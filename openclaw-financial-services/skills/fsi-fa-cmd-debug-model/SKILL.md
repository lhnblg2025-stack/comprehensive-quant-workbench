---
name: fsi-fa-cmd-debug-model
description: "Debug and audit a financial model for errors — usage: /debug-model [path to .xlsx model file]"
user-invocable: true
metadata: {}
---

Load the `check-model` skill and audit the specified financial model for broken formulas, balance sheet imbalances, hardcoded overrides, circular references, and logic errors.

If a file path is provided, use it. Otherwise ask the user for the model to review.
