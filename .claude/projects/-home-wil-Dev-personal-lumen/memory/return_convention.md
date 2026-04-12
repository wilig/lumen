---
name: Explicit return required for non-unit functions
description: Lumen requires explicit `return` at the function body level for non-unit return types. Tail expressions are still used for if/match/block sub-expressions. This eliminates the tuple-on-new-line ambiguity and gives AI agents one clear rule.
type: feedback
---

Functions returning non-unit types MUST use `return expr` as their last statement.
Tail expressions (implicit return via the last expression) are reserved for sub-blocks:
if/else branches, match arms, block expressions.

**Why:** The implicit return model caused a parser ambiguity where tuple literals on a
new line were parsed as function calls. Rather than heuristic fixes, the user chose the
explicit rule for maximum clarity for AI agents.

**How to apply:** When generating Lumen function bodies that return a value, always end
with `return expr`. For if/match expressions used as values within a function, keep
using tail expressions (the last expression in each branch).
