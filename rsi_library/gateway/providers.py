"""Operator-configured model providers and conservative per-call price ceilings."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class ProviderError(RuntimeError):
    """A provider response could not be verified."""


def _amount(tokens_in: int, tokens_out: int, input_rate: int, output_rate: int) -> int:
    return (tokens_in * input_rate + tokens_out * output_rate + 999_999) // 1_000_000


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


@dataclass(frozen=True)
class Completion:
    text: str | None
    cost_microusd: int
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    base_url: str
    model: str
    api_key_env: str
    max_input_tokens: int
    max_output_tokens: int
    input_microusd_per_million: int
    output_microusd_per_million: int
    output_limit_field: str = "max_completion_tokens"
    timeout_seconds: int = 30
    demo: bool = False
    retention: str = "unspecified"

    @property
    def reservation(self) -> int:
        return _amount(
            self.max_input_tokens, self.max_output_tokens,
            self.input_microusd_per_million, self.output_microusd_per_million,
        )

    @property
    def fingerprint(self) -> str:
        data = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(data).hexdigest()

    def public(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "max_call_cost_microusd": self.reservation,
            "demo": self.demo,
            "retention": self.retention,
        }

    def available(self) -> bool:
        return self.demo or bool(os.environ.get(self.api_key_env))

    def check_input(self, messages: list[dict[str, str]]) -> None:
        if self.demo:
            return
        if not isinstance(messages, list) or not 1 <= len(messages) <= 16:
            raise ProviderError("A model call needs between 1 and 16 messages.")
        byte_bound = 256
        for message in messages:
            if (
                not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] not in ("system", "user", "assistant")
                or not isinstance(message["content"], str)
            ):
                raise ProviderError("The optimizer submitted an invalid model message.")
            byte_bound += 64 + len(message["content"].encode("utf-8"))
        if byte_bound > self.max_input_tokens:
            raise ProviderError("The request exceeds the configured input ceiling before dispatch.")

    def complete(self, stage: str, messages: list[dict[str, str]], timeout: float, *, dispatch_id: str = "") -> Completion:
        if self.demo:
            if stage == "propose":
                evidence = json.loads(messages[-1]["content"])
                artifact = evidence["parents"][0]["artifact"]
                prompt = "Classify tickets about invoices, refunds, or charges as billing. Classify login, profile, or password tickets as account. Return only the label."
                if artifact["type"] == "prompt-chain":
                    steps = [*artifact["steps"][:-1], {**artifact["steps"][-1], "text": prompt}]
                    result = json.dumps({"steps": steps})
                elif '"improver"' in messages[0]["content"]:
                    result = json.dumps({"text": prompt, "improver": "Look for reliable topic cues, then return only the corresponding label."})
                else:
                    result = json.dumps({"text": prompt})
            elif stage.endswith("_judge"):
                graded = json.loads(messages[-1]["content"])
                same = " ".join(graded["reply"].split()).casefold() == " ".join(graded["expected"].split()).casefold()
                result = "10" if same else "0"
            else:
                text = messages[-1]["content"].casefold()
                instructions = messages[0]["content"].casefold()
                billing = any(word in text for word in ("invoice", "refund", "charge", "billing"))
                result = "billing" if "billing" in instructions and billing else "account"
            return Completion(result, 0, 0, 0)

        secret = os.environ.get(self.api_key_env)
        if not secret:
            raise ProviderError("The selected model credential is unavailable.")
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            self.output_limit_field: self.max_output_tokens,
        }).encode("utf-8")
        headers = {"Authorization": f"Bearer {secret}", "Content-Type": "application/json"}
        if dispatch_id:
            headers["X-Client-Request-Id"] = dispatch_id
        request = Request(
            self.base_url.rstrip("/") + "/chat/completions", data=body,
            headers=headers,
            method="POST",
        )
        opener = build_opener(ProxyHandler({}), _NoRedirect())
        deadline = time.monotonic() + timeout
        try:
            with opener.open(request, timeout=timeout) as response:
                chunks = bytearray()
                while chunk := response.read1(65536):
                    chunks.extend(chunk)
                    if len(chunks) > 512_000 or time.monotonic() >= deadline:
                        raise ProviderError("The model response exceeded its time or size limit.")
            payload = json.loads(chunks)
            usage = payload["usage"]
            tokens_in, tokens_out = usage["prompt_tokens"], usage["completion_tokens"]
            if any(type(count) is not int or not 0 <= count <= 100_000_000 for count in (tokens_in, tokens_out)):
                raise ProviderError("The model provider returned invalid token usage.")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ProviderError("The model call did not return verifiable usage.") from exc
        amount = _amount(tokens_in, tokens_out, self.input_microusd_per_million, self.output_microusd_per_million)
        try:
            text = payload["choices"][0]["message"]["content"]
        except (IndexError, KeyError, TypeError):
            text = None
        if not isinstance(text, str):
            text = None
        return Completion(text, amount, tokens_in, tokens_out)


def _positive(value: object, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def load_models(config_path: str | None, *, config_json: str | None = None, demo: bool = False, allow_loopback: bool = False) -> dict[str, Model]:
    models = {}
    if demo:
        models["demo"] = Model("demo", "Local demo", "", "demo", "", 1, 1, 0, 0, demo=True, retention="zero-retention")
    if config_json and config_path:
        raise ValueError("Configure models using a file or environment JSON.")
    if not config_path and not config_json:
        return models
    raw = json.loads(config_json if config_json else Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
        raise ValueError("The model configuration needs a 'models' array.")
    for entry in raw["models"]:
        if not isinstance(entry, dict) or "demo" in entry:
            raise ValueError("Each model configuration must be an object.")
        try:
            model = Model(**entry)
        except TypeError as exc:
            raise ValueError("Invalid model configuration fields.") from exc
        if (
            not isinstance(model.id, str) or not model.id.isascii() or not model.id.replace("-", "").replace("_", "").isalnum()
            or len(model.id) > 64 or model.id in models
            or not all(isinstance(value, str) and value for value in (model.name, model.model, model.api_key_env, model.base_url))
        ):
            raise ValueError("Models need a unique ID, name, endpoint, model ID, and credential variable.")
        url = urlsplit(model.base_url)
        local = allow_loopback and url.scheme == "http" and url.hostname in {"127.0.0.1", "localhost"}
        if not (url.scheme == "https" or local) or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("A model base URL must be a trusted HTTPS endpoint with no embedded credentials.")
        if not (
            _positive(model.max_input_tokens, 1, 2_000_000)
            and _positive(model.max_output_tokens, 1, 64_000)
            and _positive(model.input_microusd_per_million, 0, 2_000_000_000)
            and _positive(model.output_microusd_per_million, 0, 2_000_000_000)
            and _positive(model.timeout_seconds, 1, 120)
            and model.output_limit_field in ("max_completion_tokens", "max_tokens")
            and model.retention in ("unspecified", "zero-retention")
        ):
            raise ValueError("A model needs valid token ceilings, pricing, and a bounded request timeout.")
        models[model.id] = model
    return models
