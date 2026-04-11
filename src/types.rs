//! Bidirectional, monomorphic type checker. See `lumen-6qz`.
//!
//! Rules (per the design doc):
//! - Every binding has a knowable type (annotated or trivially inferred).
//! - No implicit numeric conversions; all widening uses `as`.
//! - No shadowing within a scope.
//! - Struct literals must provide every field.
//! - `match` must be exhaustive over sum-type variants.
//! - `?` is only valid in a fn returning a matching `Result<_, E>` / `Option<_>`.
//! - Effect checking: `pure` fns cannot call `io` fns.
//! - `string + string` is special-cased to `string.concat(a, b)`.
