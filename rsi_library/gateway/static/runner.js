"use strict";

// Runs one function from a code loop. The page starts a fresh worker for each call and
// stops it after a time limit. This script is served with its own security policy, which
// lets it evaluate code but reach nothing: no network, no page, no cookies.
const ENTRY = /^[A-Za-z_][A-Za-z0-9_]{0,63}$/;

self.onmessage = async (event) => {
  const {source, entry, input} = event.data || {};
  let output;
  try {
    if (typeof source !== "string" || typeof input !== "string" || !ENTRY.test(entry || "")) {
      throw new Error("The work item is not a function call.");
    }
    const find = new Function(`"use strict";\n${source}\n;return typeof ${entry} === "function" ? ${entry} : undefined;`);
    const fn = find();
    if (typeof fn !== "function") throw new Error(`${entry} is not defined as a function`);
    const value = await fn(input);
    output = typeof value === "string" ? value : JSON.stringify(value === undefined ? null : value);
    if (output === undefined) output = "null";
  } catch (error) {
    output = `Error: ${error && typeof error.message === "string" ? error.message : String(error)}`;
  }
  self.postMessage(String(output).slice(0, 16000));
};
