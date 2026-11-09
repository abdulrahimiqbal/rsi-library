"""Use your own agent or model client in a prompt improvement loop."""

from __future__ import annotations

from collections.abc import Callable

from .engine import ConfigError, JSONValue, Proposal, ProposalContext, TargetContext


class PromptTarget:
    """Treat the candidate as a system prompt and hand each input to your runner."""

    def __init__(self, predict: Callable[[str, JSONValue], JSONValue], initial_prompt: str):
        self._predict = predict
        self._initial_prompt = initial_prompt

    def start(self, context: TargetContext) -> str:
        return self._initial_prompt

    def validate(self, candidate: JSONValue, context: TargetContext) -> None:
        if not isinstance(candidate, str) or not candidate.strip() or len(candidate) > 32_000:
            raise ConfigError("A prompt candidate must contain 1 to 32,000 characters of text.")

    def predict(self, candidate: JSONValue, input: JSONValue, context: TargetContext) -> JSONValue:
        self.validate(candidate, context)
        assert isinstance(candidate, str)
        return self._predict(candidate, input)


class PromptEditor:
    """Call your prompt editor with the chosen parent's training feedback."""

    def __init__(self, revise: Callable[[str, ProposalContext], str]):
        self._revise = revise

    def __call__(self, context: ProposalContext) -> Proposal:
        current = context.parent.candidate
        if not isinstance(current, str):
            raise ConfigError("The prompt editor requires a text prompt as its candidate.")
        return Proposal(self._revise(current, context), note="Revised prompt using training feedback.")
