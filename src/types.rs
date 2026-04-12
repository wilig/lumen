//! Bidirectional, monomorphic type checker. See `lumen-6qz`.
//!
//! ## Rules
//!
//! - Every binding has a knowable type (annotated or trivially inferred from
//!   a literal or call).
//! - No implicit numeric conversions: `i32 → i64` requires an explicit `as`.
//! - No shadowing within a scope *or* in a nested scope — reusing a name in
//!   the same function is an error.
//! - Struct literals must provide every field.
//! - `match` must be exhaustive over sum-type variants.
//! - `?` is only valid in a function whose return type is `Result<_, E>` or
//!   `Option<_>` with a matching error/none arm.
//! - `string + string` is a typechecker special case that desugars to a
//!   `string.concat(a, b)` call.
//! - `var` bindings are mutable; `let` bindings are not. Assignment to a
//!   `let` binding or to an undeclared name is an error.
//!
//! ## Scope
//!
//! This is the MVP typechecker. It does not yet resolve stdlib function
//! calls (those land with `lumen-l64`), does not enforce pure/io effect
//! checking (that's `lumen-7xd`), and does not yet run a general iterator
//! protocol over for-loops (that's `lumen-w0g`). For-loops currently accept
//! any `List<T>` as the iterator and bind the loop variable to `T`; the
//! pseudo-function `range(i32, i32) -> List<i32>` is pre-declared as the
//! initial built-in iterator source so simple counting loops typecheck.

use std::collections::HashMap;

use crate::ast::*;
use crate::lexer::IntSuffix;
use crate::span::Span;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A resolved Lumen type. Produced by [`resolve_type`] from an AST [`Type`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ty {
    I32,
    I64,
    U32,
    U64,
    F64,
    Bool,
    String,
    Unit,
    Option(Box<Ty>),
    Result(Box<Ty>, Box<Ty>),
    List(Box<Ty>),
    /// Raw byte buffer. Same memory layout as String but without the
    /// UTF-8 assumption. Conversions are zero-cost.
    Bytes,
    /// Anonymous product type: `(i32, string)`, `(Counter, i32)`, etc.
    Tuple(Vec<Ty>),
    /// A user-declared struct or sum type by name.
    User(String),
    /// An actor handle.
    Handle(Box<Ty>),
    /// Internal placeholder emitted after a type error. Any comparison
    /// against `Error` silently succeeds so one failure doesn't cascade
    /// into a storm of follow-on errors.
    Error,
}

impl Ty {
    pub fn display(&self) -> String {
        match self {
            Ty::I32 => "i32".into(),
            Ty::I64 => "i64".into(),
            Ty::U32 => "u32".into(),
            Ty::U64 => "u64".into(),
            Ty::F64 => "f64".into(),
            Ty::Bool => "bool".into(),
            Ty::String => "string".into(),
            Ty::Bytes => "bytes".into(),
            Ty::Tuple(elems) => {
                let inner: Vec<String> = elems.iter().map(|t| t.display()).collect();
                format!("({})", inner.join(", "))
            }
            Ty::Unit => "unit".into(),
            Ty::Option(t) => format!("Option<{}>", t.display()),
            Ty::Result(o, e) => format!("Result<{}, {}>", o.display(), e.display()),
            Ty::List(t) => format!("List<{}>", t.display()),
            Ty::User(name) => name.clone(),
            Ty::Handle(inner) => format!("Handle<{}>", inner.display()),
            Ty::Error => "<error>".into(),
        }
    }

    fn is_numeric(&self) -> bool {
        matches!(
            self,
            Ty::I32 | Ty::I64 | Ty::U32 | Ty::U64 | Ty::F64
        )
    }

}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TypeError {
    pub span: Span,
    pub message: String,
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}: {}", self.span.line, self.span.col, self.message)
    }
}

impl std::error::Error for TypeError {}

// ---------------------------------------------------------------------------
// Module-level info
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct ModuleInfo {
    pub types: HashMap<String, TypeInfo>,
    pub fns: HashMap<String, FnSig>,
    /// Actor types: actor_name → list of message handler signatures.
    pub actors: HashMap<String, Vec<MsgSig>>,
}

/// Signature of an actor message handler.
#[derive(Debug, Clone)]
pub struct MsgSig {
    pub name: String,
    pub params: Vec<(String, Ty)>,
    pub ret: Ty,
}

#[derive(Debug)]
pub enum TypeInfo {
    Struct {
        fields: Vec<(String, Ty)>,
        #[allow(dead_code)]
        span: Span,
    },
    Sum {
        variants: Vec<VariantInfo>,
        #[allow(dead_code)]
        span: Span,
    },
}

#[derive(Debug)]
pub struct VariantInfo {
    pub name: String,
    pub payload: Option<VariantPayloadInfo>,
}

#[derive(Debug)]
pub enum VariantPayloadInfo {
    Named(Vec<(String, Ty)>),
    Positional(Vec<Ty>),
}

#[derive(Debug)]
pub struct FnSig {
    pub params: Vec<(String, Ty)>,
    pub ret: Ty,
    #[allow(dead_code)]
    pub effect: Effect,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn typecheck(module: &Module) -> Result<ModuleInfo, Vec<TypeError>> {
    let mut errors = Vec::new();
    let mut info = ModuleInfo {
        types: HashMap::new(),
        fns: HashMap::new(),
        actors: HashMap::new(),
    };

    // Pass 1: register all type decl names so the next pass can resolve
    // references between user types in any order.
    let mut pending_types: Vec<&TypeDecl> = Vec::new();
    for item in &module.items {
        // Register actor types as structs (pass 1 placeholder).
        if let Item::Actor(ad) = item {
            if !info.types.contains_key(&ad.name) {
                info.types.insert(
                    ad.name.clone(),
                    TypeInfo::Struct {
                        fields: Vec::new(),
                        span: ad.span,
                    },
                );
                info.actors.insert(ad.name.clone(), Vec::new());
            }
        }
        if let Item::Type(td) = item {
            if info.types.contains_key(&td.name) {
                errors.push(TypeError {
                    span: td.name_span,
                    message: format!("duplicate type `{}`", td.name),
                });
                continue;
            }
            // Insert a placeholder so later resolution can see the name.
            info.types.insert(
                td.name.clone(),
                TypeInfo::Struct {
                    fields: Vec::new(),
                    span: td.span,
                },
            );
            pending_types.push(td);
        }
    }

    // Pass 2: resolve type bodies now that names are in scope.
    for td in &pending_types {
        let body = match &td.body {
            TypeBody::Struct(fields) => {
                let mut resolved = Vec::new();
                for f in fields {
                    match resolve_type(&f.ty, &info.types) {
                        Ok(t) => resolved.push((f.name.clone(), t)),
                        Err(e) => errors.push(e),
                    }
                }
                TypeInfo::Struct {
                    fields: resolved,
                    span: td.span,
                }
            }
            TypeBody::Sum(variants) => {
                let mut out = Vec::new();
                for v in variants {
                    let payload = match &v.payload {
                        None => None,
                        Some(VariantPayload::Named(fields)) => {
                            let mut resolved = Vec::new();
                            for f in fields {
                                match resolve_type(&f.ty, &info.types) {
                                    Ok(t) => resolved.push((f.name.clone(), t)),
                                    Err(e) => errors.push(e),
                                }
                            }
                            Some(VariantPayloadInfo::Named(resolved))
                        }
                        Some(VariantPayload::Positional(tys)) => {
                            let mut resolved = Vec::new();
                            for t in tys {
                                match resolve_type(t, &info.types) {
                                    Ok(t) => resolved.push(t),
                                    Err(e) => errors.push(e),
                                }
                            }
                            Some(VariantPayloadInfo::Positional(resolved))
                        }
                    };
                    out.push(VariantInfo {
                        name: v.name.clone(),
                        payload,
                    });
                }
                TypeInfo::Sum {
                    variants: out,
                    span: td.span,
                }
            }
        };
        info.types.insert(td.name.clone(), body);
    }

    // Pass 2b: resolve actor type bodies.
    for item in &module.items {
        if let Item::Actor(ad) = item {
            let mut resolved = Vec::new();
            for f in &ad.fields {
                match resolve_type(&f.ty, &info.types) {
                    Ok(t) => resolved.push((f.name.clone(), t)),
                    Err(e) => errors.push(e),
                }
            }
            info.types.insert(
                ad.name.clone(),
                TypeInfo::Struct {
                    fields: resolved,
                    span: ad.span,
                },
            );
        }
    }

    // Pass 2c: collect msg handler signatures.
    for item in &module.items {
        if let Item::MsgHandler(mh) = item {
            let mut params = Vec::new();
            for p in &mh.params {
                match resolve_type(&p.ty, &info.types) {
                    Ok(t) => params.push((p.name.clone(), t)),
                    Err(e) => errors.push(e),
                }
            }
            let ret = match resolve_type(&mh.return_type, &info.types) {
                Ok(t) => t,
                Err(e) => {
                    errors.push(e);
                    Ty::Error
                }
            };
            // Also register as a regular fn (for codegen) with self param prepended.
            let mut fn_params = vec![("self".to_string(), Ty::User(mh.actor_name.clone()))];
            fn_params.extend(params.clone());
            let fn_name = format!("{}_{}", mh.actor_name, mh.name);
            info.fns.insert(
                fn_name,
                FnSig {
                    params: fn_params,
                    ret: ret.clone(),
                    effect: Effect::Pure,
                },
            );
            if let Some(msgs) = info.actors.get_mut(&mh.actor_name) {
                msgs.push(MsgSig {
                    name: mh.name.clone(),
                    params,
                    ret,
                });
            }
        }
    }

    // Pass 3: collect fn signatures.
    for item in &module.items {
        if let Item::Fn(f) = item {
            if info.fns.contains_key(&f.name) {
                errors.push(TypeError {
                    span: f.name_span,
                    message: format!("duplicate function `{}`", f.name),
                });
                continue;
            }
            let mut params = Vec::new();
            for p in &f.params {
                match resolve_type(&p.ty, &info.types) {
                    Ok(t) => params.push((p.name.clone(), t)),
                    Err(e) => errors.push(e),
                }
            }
            let ret = match resolve_type(&f.return_type, &info.types) {
                Ok(t) => t,
                Err(e) => {
                    errors.push(e);
                    Ty::Error
                }
            };
            info.fns.insert(
                f.name.clone(),
                FnSig {
                    params,
                    ret,
                    effect: f.effect,
                },
            );
        }
    }

    // Pass 3b: collect extern fn signatures (auto-marked io).
    for item in &module.items {
        if let Item::ExternFn(ef) = item {
            if info.fns.contains_key(&ef.name) {
                errors.push(TypeError {
                    span: ef.name_span,
                    message: format!("duplicate function `{}`", ef.name),
                });
                continue;
            }
            let mut params = Vec::new();
            for p in &ef.params {
                match resolve_type(&p.ty, &info.types) {
                    Ok(t) => params.push((p.name.clone(), t)),
                    Err(e) => errors.push(e),
                }
            }
            let ret = match resolve_type(&ef.return_type, &info.types) {
                Ok(t) => t,
                Err(e) => {
                    errors.push(e);
                    Ty::Error
                }
            };
            info.fns.insert(
                ef.name.clone(),
                FnSig {
                    params,
                    ret,
                    effect: Effect::Io,
                },
            );
        }
    }

    // Pass 3d: check msg handler bodies.
    for item in &module.items {
        if let Item::MsgHandler(mh) = item {
            let fn_name = format!("{}_{}", mh.actor_name, mh.name);
            let Some(sig) = info.fns.get(&fn_name) else {
                continue;
            };
            let mut checker = FnChecker::new(&info, sig, &mut errors);
            // Wrap body in a synthetic FnDecl for check_fn.
            let synthetic = FnDecl {
                name: fn_name.clone(),
                name_span: mh.name_span,
                params: {
                    let mut ps = vec![Param {
                        name: "self".to_string(),
                        ty: Type {
                            kind: TypeKind::Named {
                                name: mh.actor_name.clone(),
                                args: Vec::new(),
                            },
                            span: mh.name_span,
                        },
                        span: mh.name_span,
                    }];
                    ps.extend(mh.params.clone());
                    ps
                },
                return_type: mh.return_type.clone(),
                effect: Effect::Pure,
                body: mh.body.clone(),
                span: mh.span,
            };
            checker.check_fn(&synthetic);
        }
    }

    // Pass 4: check each fn body.
    for item in &module.items {
        if let Item::Fn(f) = item {
            let Some(sig) = info.fns.get(&f.name) else {
                continue;
            };
            let mut checker = FnChecker::new(&info, sig, &mut errors);
            checker.check_fn(f);
        }
    }

    if errors.is_empty() {
        Ok(info)
    } else {
        Err(errors)
    }
}

// ---------------------------------------------------------------------------
// Type resolution
// ---------------------------------------------------------------------------

fn resolve_type(t: &Type, types: &HashMap<String, TypeInfo>) -> Result<Ty, TypeError> {
    match &t.kind {
        TypeKind::Named { name, args } => match name.as_str() {
            "i32" if args.is_empty() => Ok(Ty::I32),
            "i64" if args.is_empty() => Ok(Ty::I64),
            "u32" if args.is_empty() => Ok(Ty::U32),
            "u64" if args.is_empty() => Ok(Ty::U64),
            "f64" if args.is_empty() => Ok(Ty::F64),
            "bool" if args.is_empty() => Ok(Ty::Bool),
            "string" if args.is_empty() => Ok(Ty::String),
            "bytes" if args.is_empty() => Ok(Ty::Bytes),
            "unit" if args.is_empty() => Ok(Ty::Unit),
            "Option" if args.len() == 1 => {
                Ok(Ty::Option(Box::new(resolve_type(&args[0], types)?)))
            }
            "Result" if args.len() == 2 => Ok(Ty::Result(
                Box::new(resolve_type(&args[0], types)?),
                Box::new(resolve_type(&args[1], types)?),
            )),
            "List" if args.len() == 1 => {
                Ok(Ty::List(Box::new(resolve_type(&args[0], types)?)))
            }
            "Handle" if args.len() == 1 => {
                Ok(Ty::Handle(Box::new(resolve_type(&args[0], types)?)))
            }
            _ if args.is_empty() && types.contains_key(name) => Ok(Ty::User(name.clone())),
            _ => Err(TypeError {
                span: t.span,
                message: format!("unknown type `{name}`"),
            }),
        },
        TypeKind::Tuple(elems) => {
            let resolved: Result<Vec<Ty>, TypeError> =
                elems.iter().map(|e| resolve_type(e, types)).collect();
            Ok(Ty::Tuple(resolved?))
        }
    }
}

// ---------------------------------------------------------------------------
// Function body checker
// ---------------------------------------------------------------------------

struct FnChecker<'a> {
    module: &'a ModuleInfo,
    sig: &'a FnSig,
    scopes: Vec<Scope>,
    errors: &'a mut Vec<TypeError>,
}

#[derive(Default)]
struct Scope {
    bindings: HashMap<String, Binding>,
}

#[derive(Clone)]
struct Binding {
    ty: Ty,
    mutable: bool,
    declared_at: Span,
}

impl<'a> FnChecker<'a> {
    fn new(module: &'a ModuleInfo, sig: &'a FnSig, errors: &'a mut Vec<TypeError>) -> Self {
        Self {
            module,
            sig,
            scopes: Vec::new(),
            errors,
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(Scope::default());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn declare(&mut self, name: &str, ty: Ty, mutable: bool, span: Span) {
        // No shadowing: a name declared in any currently-live scope can't
        // be reused. Sibling scopes (e.g., separate match arms) don't
        // overlap, so they can freely reuse the same name.
        if let Some(prior) = self.lookup(name) {
            self.errors.push(TypeError {
                span,
                message: format!(
                    "`{name}` is already declared at {}:{}; shadowing is not allowed",
                    prior.declared_at.line, prior.declared_at.col
                ),
            });
        }
        self.scopes.last_mut().unwrap().bindings.insert(
            name.to_string(),
            Binding {
                ty,
                mutable,
                declared_at: span,
            },
        );
    }

    fn lookup(&self, name: &str) -> Option<&Binding> {
        for scope in self.scopes.iter().rev() {
            if let Some(b) = scope.bindings.get(name) {
                return Some(b);
            }
        }
        None
    }

    fn check_fn(&mut self, f: &FnDecl) {
        self.push_scope();
        for (pname, pty) in &self.sig.params {
            // Pre-existing guarantee: params are unique names at parse time,
            // but we still record them through declare() so shadowing checks
            // work against them.
            self.declare(pname, pty.clone(), false, f.name_span);
        }
        let ret = self.sig.ret.clone();
        let body_ty = self.check_block(&f.body, Some(&ret));
        let ends_with_return = f
            .body
            .stmts
            .last()
            .is_some_and(|s| matches!(s.kind, StmtKind::Return(_)));
        // Non-unit functions MUST use explicit `return`. Tail expressions
        // are reserved for sub-blocks (if/match/closures), not function
        // bodies. This eliminates the tuple-on-new-line ambiguity and
        // gives AI agents one clear rule.
        if !matches!(ret, Ty::Unit | Ty::Error) && !ends_with_return {
            self.errors.push(TypeError {
                span: f.body.span,
                message: format!(
                    "function `{}` returns {} — use `return expr` (explicit return required for non-unit functions)",
                    f.name,
                    ret.display()
                ),
            });
        }
        self.pop_scope();
    }

    fn check_block(&mut self, block: &Block, expected: Option<&Ty>) -> Ty {
        self.push_scope();
        for stmt in &block.stmts {
            self.check_stmt(stmt);
        }
        let ty = match &block.tail {
            Some(e) => match expected {
                Some(t) => self.check_expr(e, t),
                None => self.infer_expr(e),
            },
            None => Ty::Unit,
        };
        self.pop_scope();
        ty
    }

    fn check_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Let { name, ty, value } => {
                let bound_ty = match ty {
                    Some(ann) => match resolve_type(ann, &self.module.types) {
                        Ok(t) => {
                            self.check_expr(value, &t);
                            t
                        }
                        Err(e) => {
                            self.errors.push(e);
                            Ty::Error
                        }
                    },
                    None => self.infer_expr(value),
                };
                self.declare(name, bound_ty, false, stmt.span);
            }
            StmtKind::Var { name, ty, value } => {
                let bound_ty = match ty {
                    Some(ann) => match resolve_type(ann, &self.module.types) {
                        Ok(t) => {
                            self.check_expr(value, &t);
                            t
                        }
                        Err(e) => {
                            self.errors.push(e);
                            Ty::Error
                        }
                    },
                    None => self.infer_expr(value),
                };
                self.declare(name, bound_ty, true, stmt.span);
            }
            StmtKind::Assign { name, value } => {
                match self.lookup(name).cloned() {
                    None => {
                        self.errors.push(TypeError {
                            span: stmt.span,
                            message: format!("assignment to undeclared name `{name}`"),
                        });
                        // Still infer the RHS so errors inside it get reported.
                        self.infer_expr(value);
                    }
                    Some(b) if !b.mutable => {
                        self.errors.push(TypeError {
                            span: stmt.span,
                            message: format!(
                                "cannot assign to `{name}`: it was declared with `let`, use `var` for mutable bindings"
                            ),
                        });
                        self.infer_expr(value);
                    }
                    Some(b) => {
                        self.check_expr(value, &b.ty);
                    }
                }
            }
            StmtKind::Expr(e) => {
                self.infer_expr(e);
            }
            StmtKind::For { binder, iter, body } => {
                let iter_ty = self.infer_expr(iter);
                let elem_ty = match &iter_ty {
                    Ty::List(t) => (**t).clone(),
                    Ty::Error => Ty::Error,
                    other => {
                        self.errors.push(TypeError {
                            span: iter.span,
                            message: format!(
                                "for-loop iterator must be a List<T>, found {}",
                                other.display()
                            ),
                        });
                        Ty::Error
                    }
                };
                self.push_scope();
                self.declare(binder, elem_ty, false, stmt.span);
                self.check_block(body, Some(&Ty::Unit));
                self.pop_scope();
            }
            StmtKind::LetTuple { names, value } => {
                let val_ty = self.infer_expr(value);
                match val_ty {
                    Ty::Tuple(ref elems) => {
                        if names.len() != elems.len() {
                            self.errors.push(TypeError {
                                span: stmt.span,
                                message: format!(
                                    "tuple destructuring expects {} names, found {}",
                                    elems.len(),
                                    names.len()
                                ),
                            });
                        } else {
                            for (i, name) in names.iter().enumerate() {
                                let elem_ty = elems[i].clone();
                                self.declare(name, elem_ty, false, stmt.span);
                            }
                        }
                    }
                    Ty::Error => {}
                    other => {
                        self.errors.push(TypeError {
                            span: stmt.span,
                            message: format!(
                                "`let (...)` destructuring requires a tuple type, found {}",
                                other.display()
                            ),
                        });
                    }
                }
            }
            StmtKind::Return(value) => match value {
                None => {
                    if !compatible(&Ty::Unit, &self.sig.ret) {
                        self.errors.push(TypeError {
                            span: stmt.span,
                            message: format!(
                                "`return` without a value in a function returning {}",
                                self.sig.ret.display()
                            ),
                        });
                    }
                }
                Some(e) => {
                    let ret = self.sig.ret.clone();
                    self.check_expr(e, &ret);
                }
            },
        }
    }

    fn check_expr(&mut self, expr: &Expr, expected: &Ty) -> Ty {
        let actual = self.infer_expr(expr);
        if !compatible(&actual, expected) {
            self.errors.push(TypeError {
                span: expr.span,
                message: format!(
                    "type mismatch: expected {}, found {}",
                    expected.display(),
                    actual.display()
                ),
            });
        }
        actual
    }

    fn infer_expr(&mut self, expr: &Expr) -> Ty {
        match &expr.kind {
            ExprKind::IntLit { suffix, .. } => match suffix {
                Some(IntSuffix::I32) => Ty::I32,
                Some(IntSuffix::I64) => Ty::I64,
                Some(IntSuffix::U32) => Ty::U32,
                Some(IntSuffix::U64) => Ty::U64,
                None => Ty::I32,
            },
            ExprKind::FloatLit(_) => Ty::F64,
            ExprKind::StringLit(_) => Ty::String,
            ExprKind::BoolLit(_) => Ty::Bool,
            ExprKind::UnitLit => Ty::Unit,

            ExprKind::Ident(name) => {
                if let Some(b) = self.lookup(name) {
                    b.ty.clone()
                } else if name == "None" {
                    // Bare None constructor — its argument-free nature means
                    // we can't infer T here. If called in a check context,
                    // `check_expr` will accept it against an Option<T>.
                    Ty::Option(Box::new(Ty::Error))
                } else if let Some((ty_name, variant)) = self.find_variant(name) {
                    // Bare zero-payload variant constructor, e.g.
                    // `type T = | A | B` called as `A`.
                    if variant.payload.is_none() {
                        Ty::User(ty_name)
                    } else {
                        self.errors.push(TypeError {
                            span: expr.span,
                            message: format!(
                                "variant `{name}` has a payload; call it with arguments"
                            ),
                        });
                        Ty::Error
                    }
                } else {
                    self.errors.push(TypeError {
                        span: expr.span,
                        message: format!("unknown identifier `{name}`"),
                    });
                    Ty::Error
                }
            }

            ExprKind::Paren(inner) => self.infer_expr(inner),

            ExprKind::Block(block) => self.check_block(block, None),

            ExprKind::StructLit {
                name,
                name_span,
                fields,
                spread,
            } => self.check_struct_lit(name, *name_span, fields, spread.as_deref(), expr.span),

            ExprKind::Call { callee, args } => self.check_call(callee, args, expr.span),

            ExprKind::Field { receiver, name } => {
                let recv = self.infer_expr(receiver);
                self.check_field_access(&recv, name, expr.span)
            }

            ExprKind::MethodCall { receiver, method, args } => {
                // Recognize module-qualified calls BEFORE inferring the
                // receiver type, because module names (`io`, `int`, ...)
                // are not values and would fail type inference.
                if let ExprKind::Ident(mod_name) = &receiver.kind {
                    if let Some(ret) = self.check_module_call(
                        mod_name, method, args, expr.span,
                    ) {
                        return ret;
                    }
                }
                let _ = self.infer_expr(receiver);
                self.errors.push(TypeError {
                    span: expr.span,
                    message: format!(
                        "method `.{method}(...)` not yet supported by the typechecker (stdlib resolution lands with lumen-l64)"
                    ),
                });
                Ty::Error
            }

            ExprKind::Try(inner) => self.check_try(inner, expr.span),

            ExprKind::Unary { op, rhs } => {
                let rhs_ty = self.infer_expr(rhs);
                match op {
                    UnaryOp::Neg => {
                        if rhs_ty.is_numeric() || matches!(rhs_ty, Ty::Error) {
                            rhs_ty
                        } else {
                            self.errors.push(TypeError {
                                span: expr.span,
                                message: format!(
                                    "unary `-` expects a numeric type, found {}",
                                    rhs_ty.display()
                                ),
                            });
                            Ty::Error
                        }
                    }
                    UnaryOp::Not => {
                        if matches!(rhs_ty, Ty::Bool | Ty::Error) {
                            Ty::Bool
                        } else {
                            self.errors.push(TypeError {
                                span: expr.span,
                                message: format!(
                                    "unary `!` expects a bool, found {}",
                                    rhs_ty.display()
                                ),
                            });
                            Ty::Error
                        }
                    }
                }
            }

            ExprKind::Binary { op, lhs, rhs } => self.check_binary(*op, lhs, rhs, expr.span),

            ExprKind::Cast { expr: inner, to } => {
                let from = self.infer_expr(inner);
                let to_ty = match resolve_type(to, &self.module.types) {
                    Ok(t) => t,
                    Err(e) => {
                        self.errors.push(e);
                        return Ty::Error;
                    }
                };
                if !from.is_numeric() && !matches!(from, Ty::Error) {
                    self.errors.push(TypeError {
                        span: expr.span,
                        message: format!(
                            "`as` cast requires a numeric source, found {}",
                            from.display()
                        ),
                    });
                }
                if !to_ty.is_numeric() {
                    self.errors.push(TypeError {
                        span: expr.span,
                        message: format!(
                            "`as` cast requires a numeric target, found {}",
                            to_ty.display()
                        ),
                    });
                }
                to_ty
            }

            ExprKind::If {
                cond,
                then_block,
                else_block,
            } => {
                self.check_expr(cond, &Ty::Bool);
                let t = self.check_block(then_block, None);
                let e = self.check_block(else_block, Some(&t));
                if !compatible(&t, &e) {
                    self.errors.push(TypeError {
                        span: expr.span,
                        message: format!(
                            "if branches produce different types: {} vs {}",
                            t.display(),
                            e.display()
                        ),
                    });
                }
                t
            }

            ExprKind::Spawn { actor_name, fields } => {
                if !self.module.actors.contains_key(actor_name) {
                    self.errors.push(TypeError {
                        span: expr.span,
                        message: format!("`{actor_name}` is not an actor type"),
                    });
                    return Ty::Error;
                }
                // Check fields like a struct literal.
                self.check_struct_lit(actor_name, expr.span, fields, None, expr.span);
                Ty::Handle(Box::new(Ty::User(actor_name.clone())))
            }
            ExprKind::Send { handle, method, args } => {
                let handle_ty = self.infer_expr(handle);
                let Ty::Handle(inner) = &handle_ty else {
                    self.errors.push(TypeError {
                        span: handle.span,
                        message: format!(
                            "`send` requires an actor handle, found {}",
                            handle_ty.display()
                        ),
                    });
                    return Ty::Unit;
                };
                let Ty::User(actor_name) = inner.as_ref() else {
                    return Ty::Unit;
                };
                self.check_msg_args(actor_name, method, args, expr.span);
                Ty::Unit
            }
            ExprKind::Ask { handle, method, args } => {
                let handle_ty = self.infer_expr(handle);
                let Ty::Handle(inner) = &handle_ty else {
                    self.errors.push(TypeError {
                        span: handle.span,
                        message: format!(
                            "`ask` requires an actor handle, found {}",
                            handle_ty.display()
                        ),
                    });
                    return Ty::Error;
                };
                let Ty::User(actor_name) = inner.as_ref() else {
                    return Ty::Error;
                };
                self.check_msg_args(actor_name, method, args, expr.span)
            }
            ExprKind::TupleLit(elems) => {
                let types: Vec<Ty> = elems.iter().map(|e| self.infer_expr(e)).collect();
                Ty::Tuple(types)
            }

            ExprKind::TupleField { receiver, index } => {
                let recv_ty = self.infer_expr(receiver);
                match recv_ty {
                    Ty::Tuple(ref elems) => {
                        let idx = *index as usize;
                        if idx < elems.len() {
                            elems[idx].clone()
                        } else {
                            self.errors.push(TypeError {
                                span: expr.span,
                                message: format!(
                                    "tuple index {} out of bounds for tuple with {} elements",
                                    index,
                                    elems.len()
                                ),
                            });
                            Ty::Error
                        }
                    }
                    Ty::Error => Ty::Error,
                    other => {
                        self.errors.push(TypeError {
                            span: expr.span,
                            message: format!(
                                "tuple field access on a non-tuple type {}",
                                other.display()
                            ),
                        });
                        Ty::Error
                    }
                }
            }

            ExprKind::Match { scrutinee, arms } => self.check_match(scrutinee, arms, expr.span),
        }
    }

    // --- Specialized expression checks ------------------------------------

    fn check_binary(&mut self, op: BinOp, lhs: &Expr, rhs: &Expr, span: Span) -> Ty {
        let lt = self.infer_expr(lhs);
        let rt = self.infer_expr(rhs);

        // String concatenation special case.
        if matches!(op, BinOp::Add) && matches!(lt, Ty::String) && matches!(rt, Ty::String) {
            return Ty::String;
        }

        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => {
                if (lt.is_numeric() || matches!(lt, Ty::Error))
                    && (rt.is_numeric() || matches!(rt, Ty::Error))
                    && compatible(&lt, &rt)
                {
                    if matches!(lt, Ty::Error) {
                        rt
                    } else {
                        lt
                    }
                } else {
                    self.errors.push(TypeError {
                        span,
                        message: format!(
                            "arithmetic operator expects matching numeric types, found {} and {}",
                            lt.display(),
                            rt.display()
                        ),
                    });
                    Ty::Error
                }
            }
            BinOp::Eq | BinOp::NotEq => {
                if !compatible(&lt, &rt) {
                    self.errors.push(TypeError {
                        span,
                        message: format!(
                            "equality operands have different types: {} and {}",
                            lt.display(),
                            rt.display()
                        ),
                    });
                }
                Ty::Bool
            }
            BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq => {
                if !((lt.is_numeric() || matches!(lt, Ty::Error))
                    && (rt.is_numeric() || matches!(rt, Ty::Error))
                    && compatible(&lt, &rt))
                {
                    self.errors.push(TypeError {
                        span,
                        message: format!(
                            "comparison expects matching numeric types, found {} and {}",
                            lt.display(),
                            rt.display()
                        ),
                    });
                }
                Ty::Bool
            }
            BinOp::And | BinOp::Or => {
                if !matches!(lt, Ty::Bool | Ty::Error) || !matches!(rt, Ty::Bool | Ty::Error) {
                    self.errors.push(TypeError {
                        span,
                        message: format!(
                            "logical operator expects bool operands, found {} and {}",
                            lt.display(),
                            rt.display()
                        ),
                    });
                }
                Ty::Bool
            }
        }
    }

    fn check_struct_lit(
        &mut self,
        name: &str,
        name_span: Span,
        fields: &[FieldInit],
        spread: Option<&Expr>,
        whole_span: Span,
    ) -> Ty {
        // First try: a regular user struct type.
        if let Some(TypeInfo::Struct {
            fields: def_fields,
            ..
        }) = self.module.types.get(name)
        {
            let def_fields: Vec<(String, Ty)> = def_fields.clone();
            return self.check_struct_lit_fields(name, name_span, fields, spread, whole_span, def_fields);
        }

        // Second try: a sum-type variant constructor with named fields,
        // e.g. `Circle { radius: 1.0 }` for `type Shape = | Circle { ... }`.
        if let Some((sum_name, variant_def_fields)) = self.find_named_variant(name) {
            self.check_struct_lit_fields(
                name,
                name_span,
                fields,
                spread,
                whole_span,
                variant_def_fields,
            );
            return Ty::User(sum_name);
        }

        self.errors.push(TypeError {
            span: name_span,
            message: format!("`{name}` is not a struct type or named-field variant"),
        });
        Ty::Error
    }

    /// Look up `variant_name` in every user sum type, returning the
    /// owning type's name and the variant's named fields if it has that
    /// shape.
    fn find_named_variant(&self, variant_name: &str) -> Option<(String, Vec<(String, Ty)>)> {
        for (type_name, info) in &self.module.types {
            if let TypeInfo::Sum { variants, .. } = info {
                for v in variants {
                    if v.name == variant_name {
                        if let Some(VariantPayloadInfo::Named(fields)) = &v.payload {
                            return Some((type_name.clone(), fields.clone()));
                        }
                    }
                }
            }
        }
        None
    }

    fn check_struct_lit_fields(
        &mut self,
        name: &str,
        name_span: Span,
        fields: &[FieldInit],
        spread: Option<&Expr>,
        whole_span: Span,
        def_fields: Vec<(String, Ty)>,
    ) -> Ty {
        let _ = name_span;

        // If there is a spread expression, typecheck it and verify it has the
        // same struct type.  Missing fields are then provided by the spread.
        let has_spread = if let Some(spread_expr) = spread {
            let spread_ty = self.infer_expr(spread_expr);
            if !matches!(&spread_ty, Ty::User(n) if n == name) && !matches!(spread_ty, Ty::Error) {
                self.errors.push(TypeError {
                    span: spread_expr.span,
                    message: format!(
                        "spread expression has type `{}` but struct literal is `{name}`",
                        spread_ty.display()
                    ),
                });
            }
            true
        } else {
            false
        };

        // Check every provided field, collect names.
        let mut provided: HashMap<&str, &Expr> = HashMap::new();
        for fi in fields {
            if provided.insert(&fi.name, &fi.value).is_some() {
                self.errors.push(TypeError {
                    span: fi.span,
                    message: format!("field `{}` initialized twice", fi.name),
                });
            }
        }

        for (fname, fty) in &def_fields {
            match provided.remove(fname.as_str()) {
                Some(e) => {
                    self.check_expr(e, fty);
                }
                None => {
                    // If there is a spread, missing fields come from there — OK.
                    if !has_spread {
                        self.errors.push(TypeError {
                            span: whole_span,
                            message: format!(
                                "struct `{name}` is missing field `{fname}` of type {}",
                                fty.display()
                            ),
                        });
                    }
                }
            }
        }
        for unexpected in provided.keys() {
            self.errors.push(TypeError {
                span: name_span,
                message: format!("struct `{name}` has no field `{unexpected}`"),
            });
        }

        Ty::User(name.to_string())
    }

    fn check_call(&mut self, callee: &Expr, args: &[Arg], whole_span: Span) -> Ty {
        // Pull out a bare-identifier callee for built-in constructors,
        // user functions, and the stub `range` iterator source.
        if let ExprKind::Ident(name) = &callee.kind {
            // Built-in Option/Result constructors.
            match name.as_str() {
                "Ok" => return self.check_ok_call(args, whole_span),
                "Err" => return self.check_err_call(args, whole_span),
                "Some" => return self.check_some_call(args, whole_span),
                "None" => {
                    // None is not callable.
                    self.errors.push(TypeError {
                        span: callee.span,
                        message: "`None` is not callable; write `None` bare".into(),
                    });
                    return Ty::Option(Box::new(Ty::Error));
                }
                "range" => return self.check_range_call(args, whole_span),
                "string_len" => return self.check_string_len_call(args, whole_span),
                _ => {}
            }

            // User fn call.
            if let Some(sig) = self.module.fns.get(name).map(|s| FnSig {
                params: s.params.clone(),
                ret: s.ret.clone(),
                effect: s.effect,
            }) {
                self.check_effect(sig.effect, whole_span);
                self.check_args_against_params(&sig.params, args, whole_span);
                return sig.ret;
            }

            // Variant constructor? Look through all sum types.
            if let Some((ty_name, variant)) = self.find_variant(name) {
                return self.check_variant_call(&ty_name, variant, args, whole_span);
            }

            self.errors.push(TypeError {
                span: callee.span,
                message: format!("unknown function `{name}`"),
            });
            return Ty::Error;
        }

        // Anything else (e.g. calling the result of an expression) isn't
        // supported in v1.
        self.errors.push(TypeError {
            span: whole_span,
            message: "only direct function calls are supported".into(),
        });
        Ty::Error
    }

    fn check_args_against_params(
        &mut self,
        params: &[(String, Ty)],
        args: &[Arg],
        call_span: Span,
    ) {
        if args.len() != params.len() {
            self.errors.push(TypeError {
                span: call_span,
                message: format!(
                    "expected {} arguments, found {}",
                    params.len(),
                    args.len()
                ),
            });
            return;
        }
        for (i, arg) in args.iter().enumerate() {
            let (pname, pty) = &params[i];
            if let Some(ref aname) = arg.name {
                if aname != pname {
                    self.errors.push(TypeError {
                        span: arg.span,
                        message: format!(
                            "named argument `{aname}` does not match parameter `{pname}`"
                        ),
                    });
                }
            }
            self.check_expr(&arg.value, pty);
        }
    }

    fn check_ok_call(&mut self, args: &[Arg], call_span: Span) -> Ty {
        if args.len() != 1 {
            self.errors.push(TypeError {
                span: call_span,
                message: format!("`Ok` expects 1 argument, found {}", args.len()),
            });
            return Ty::Result(Box::new(Ty::Error), Box::new(Ty::Error));
        }
        let ok_ty = self.infer_expr(&args[0].value);
        Ty::Result(Box::new(ok_ty), Box::new(Ty::Error))
    }

    fn check_err_call(&mut self, args: &[Arg], call_span: Span) -> Ty {
        if args.len() != 1 {
            self.errors.push(TypeError {
                span: call_span,
                message: format!("`Err` expects 1 argument, found {}", args.len()),
            });
            return Ty::Result(Box::new(Ty::Error), Box::new(Ty::Error));
        }
        let err_ty = self.infer_expr(&args[0].value);
        Ty::Result(Box::new(Ty::Error), Box::new(err_ty))
    }

    fn check_some_call(&mut self, args: &[Arg], call_span: Span) -> Ty {
        if args.len() != 1 {
            self.errors.push(TypeError {
                span: call_span,
                message: format!("`Some` expects 1 argument, found {}", args.len()),
            });
            return Ty::Option(Box::new(Ty::Error));
        }
        let inner = self.infer_expr(&args[0].value);
        Ty::Option(Box::new(inner))
    }

    /// Recognize module-qualified calls like `io.println(s)`. Returns
    /// `Some(return_type)` if recognized, `None` to fall through to the
    /// generic method error.
    fn check_module_call(
        &mut self,
        module: &str,
        method: &str,
        args: &[Arg],
        span: Span,
    ) -> Option<Ty> {
        match (module, method) {
            ("int", "to_string_i32") => {
                if args.len() != 1 {
                    self.errors.push(TypeError {
                        span,
                        message: format!(
                            "`int.to_string_i32` expects 1 argument, found {}",
                            args.len()
                        ),
                    });
                } else {
                    self.check_expr(&args[0].value, &Ty::I32);
                }
                Some(Ty::String)
            }
            ("io", "println") => {
                self.check_effect(Effect::Io, span);
                if args.len() != 1 {
                    self.errors.push(TypeError {
                        span,
                        message: format!(
                            "`io.println` expects 1 argument, found {}",
                            args.len()
                        ),
                    });
                } else {
                    self.check_expr(&args[0].value, &Ty::String);
                }
                Some(Ty::Unit)
            }
            ("bytes", "len") => {
                if args.len() != 1 {
                    self.errors.push(TypeError {
                        span,
                        message: format!("`bytes.len` expects 1 argument, found {}", args.len()),
                    });
                } else {
                    self.check_expr(&args[0].value, &Ty::Bytes);
                }
                Some(Ty::I32)
            }
            ("bytes", "new") => {
                if args.len() != 1 {
                    self.errors.push(TypeError {
                        span,
                        message: format!("`bytes.new` expects 1 argument, found {}", args.len()),
                    });
                } else {
                    self.check_expr(&args[0].value, &Ty::I32);
                }
                Some(Ty::Bytes)
            }
            ("bytes", "get") => {
                if args.len() != 2 {
                    self.errors.push(TypeError {
                        span,
                        message: format!("`bytes.get` expects 2 arguments, found {}", args.len()),
                    });
                } else {
                    self.check_expr(&args[0].value, &Ty::Bytes);
                    self.check_expr(&args[1].value, &Ty::I32);
                }
                Some(Ty::I32)
            }
            ("bytes", "concat") => {
                if args.len() != 2 {
                    self.errors.push(TypeError {
                        span,
                        message: format!("`bytes.concat` expects 2 arguments, found {}", args.len()),
                    });
                } else {
                    self.check_expr(&args[0].value, &Ty::Bytes);
                    self.check_expr(&args[1].value, &Ty::Bytes);
                }
                Some(Ty::Bytes)
            }
            ("bytes", "from_string") => {
                if args.len() != 1 {
                    self.errors.push(TypeError {
                        span,
                        message: format!("`bytes.from_string` expects 1 argument, found {}", args.len()),
                    });
                } else {
                    self.check_expr(&args[0].value, &Ty::String);
                }
                Some(Ty::Bytes)
            }
            ("string", "from_bytes") => {
                if args.len() != 1 {
                    self.errors.push(TypeError {
                        span,
                        message: format!("`string.from_bytes` expects 1 argument, found {}", args.len()),
                    });
                } else {
                    self.check_expr(&args[0].value, &Ty::Bytes);
                }
                Some(Ty::String)
            }
            ("net", "tcp_listen") => {
                self.check_effect(Effect::Io, span);
                if args.len() != 1 {
                    self.errors.push(TypeError {
                        span,
                        message: format!("`net.tcp_listen` expects 1 argument, found {}", args.len()),
                    });
                } else {
                    self.check_expr(&args[0].value, &Ty::I32);
                }
                Some(Ty::I32)
            }
            ("net", "tcp_accept") => {
                self.check_effect(Effect::Io, span);
                if args.len() != 1 {
                    self.errors.push(TypeError {
                        span,
                        message: format!("`net.tcp_accept` expects 1 argument, found {}", args.len()),
                    });
                } else {
                    self.check_expr(&args[0].value, &Ty::I32);
                }
                Some(Ty::I32)
            }
            ("net", "tcp_read") => {
                self.check_effect(Effect::Io, span);
                if args.len() != 2 {
                    self.errors.push(TypeError {
                        span,
                        message: format!("`net.tcp_read` expects 2 arguments, found {}", args.len()),
                    });
                } else {
                    self.check_expr(&args[0].value, &Ty::I32);
                    self.check_expr(&args[1].value, &Ty::I32);
                }
                Some(Ty::Bytes)
            }
            ("net", "tcp_write") => {
                self.check_effect(Effect::Io, span);
                if args.len() != 2 {
                    self.errors.push(TypeError {
                        span,
                        message: format!("`net.tcp_write` expects 2 arguments, found {}", args.len()),
                    });
                } else {
                    self.check_expr(&args[0].value, &Ty::I32);
                    self.check_expr(&args[1].value, &Ty::Bytes);
                }
                Some(Ty::I32)
            }
            ("net", "serve") => {
                self.check_effect(Effect::Io, span);
                if args.len() != 2 {
                    self.errors.push(TypeError {
                        span,
                        message: format!("`net.serve` expects 2 arguments (port, handler), found {}", args.len()),
                    });
                } else {
                    self.check_expr(&args[0].value, &Ty::I32);
                    // Second arg is a function name — type-check it as an ident.
                    // The codegen will resolve it to a func_addr.
                }
                Some(Ty::Unit)
            }
            // --- List operations ---
            ("list", "new") => {
                Some(Ty::List(Box::new(Ty::Error)))
            }
            ("list", "len") => {
                if args.len() != 1 { self.errors.push(TypeError { span, message: "`list.len` expects 1 arg".into() }); }
                Some(Ty::I32)
            }
            ("list", "push") => {
                if args.len() != 2 { self.errors.push(TypeError { span, message: "`list.push` expects 2 args".into() }); }
                if let Some(a) = args.first() { return Some(self.infer_expr(&a.value)); }
                Some(Ty::List(Box::new(Ty::Error)))
            }
            ("list", "get") => {
                if args.len() != 2 { self.errors.push(TypeError { span, message: "`list.get` expects 2 args".into() }); }
                if let Some(a) = args.first() {
                    let lt = self.infer_expr(&a.value);
                    if let Ty::List(inner) = lt { return Some(*inner); }
                }
                Some(Ty::I32)
            }
            ("list", "set") => {
                if args.len() != 3 { self.errors.push(TypeError { span, message: "`list.set` expects 3 args".into() }); }
                if let Some(a) = args.first() { return Some(self.infer_expr(&a.value)); }
                Some(Ty::List(Box::new(Ty::Error)))
            }
            ("list", "remove") => {
                if args.len() != 2 { self.errors.push(TypeError { span, message: "`list.remove` expects 2 args".into() }); }
                if let Some(a) = args.first() { return Some(self.infer_expr(&a.value)); }
                Some(Ty::List(Box::new(Ty::Error)))
            }
            ("net", "gt_read") => {
                self.check_effect(Effect::Io, span);
                if args.len() != 2 {
                    self.errors.push(TypeError { span, message: format!("`net.gt_read` expects 2 args, found {}", args.len()) });
                } else {
                    self.check_expr(&args[0].value, &Ty::I32);
                    self.check_expr(&args[1].value, &Ty::I32);
                }
                Some(Ty::Bytes)
            }
            ("net", "gt_write") => {
                self.check_effect(Effect::Io, span);
                if args.len() != 2 {
                    self.errors.push(TypeError { span, message: format!("`net.gt_write` expects 2 args, found {}", args.len()) });
                } else {
                    self.check_expr(&args[0].value, &Ty::I32);
                    self.check_expr(&args[1].value, &Ty::Bytes);
                }
                Some(Ty::I32)
            }
            ("net", "tcp_close") => {
                self.check_effect(Effect::Io, span);
                if args.len() != 1 {
                    self.errors.push(TypeError {
                        span,
                        message: format!("`net.tcp_close` expects 1 argument, found {}", args.len()),
                    });
                } else {
                    self.check_expr(&args[0].value, &Ty::I32);
                }
                Some(Ty::Unit)
            }
            ("http", "parse_method") | ("http", "parse_path") | ("http", "parse_body") => {
                if args.len() != 1 {
                    self.errors.push(TypeError {
                        span,
                        message: format!(
                            "`http.{method}` expects 1 argument, found {}",
                            args.len()
                        ),
                    });
                } else {
                    self.check_expr(&args[0].value, &Ty::Bytes);
                }
                Some(Ty::String)
            }
            ("http", "format_response") => {
                if args.len() != 2 {
                    self.errors.push(TypeError {
                        span,
                        message: format!(
                            "`http.format_response` expects 2 arguments, found {}",
                            args.len()
                        ),
                    });
                } else {
                    self.check_expr(&args[0].value, &Ty::I32);
                    self.check_expr(&args[1].value, &Ty::String);
                }
                Some(Ty::Bytes)
            }
            _ => None,
        }
    }

    /// Check that the current function's effect allows calling a function
    /// with the given effect. A `pure` function cannot call an `io` function.
    fn check_msg_args(
        &mut self,
        actor_name: &str,
        method: &str,
        args: &[Arg],
        span: Span,
    ) -> Ty {
        let Some(msgs) = self.module.actors.get(actor_name) else {
            self.errors.push(TypeError {
                span,
                message: format!("unknown actor type `{actor_name}`"),
            });
            return Ty::Error;
        };
        let Some(msg_sig) = msgs.iter().find(|m| m.name == method) else {
            self.errors.push(TypeError {
                span,
                message: format!(
                    "actor `{actor_name}` has no message handler `{method}`"
                ),
            });
            return Ty::Error;
        };
        if args.len() != msg_sig.params.len() {
            self.errors.push(TypeError {
                span,
                message: format!(
                    "message `{method}` expects {} argument(s), found {}",
                    msg_sig.params.len(),
                    args.len()
                ),
            });
        } else {
            for (i, arg) in args.iter().enumerate() {
                let (_, pty) = &msg_sig.params[i];
                self.check_expr(&arg.value, pty);
            }
        }
        msg_sig.ret.clone()
    }

    fn check_effect(&mut self, callee_effect: Effect, span: Span) {
        if self.sig.effect == Effect::Pure && callee_effect == Effect::Io {
            self.errors.push(TypeError {
                span,
                message: "a `pure` function cannot call an `io` function".into(),
            });
        }
    }

    fn check_string_len_call(&mut self, args: &[Arg], call_span: Span) -> Ty {
        // Built-in `string_len(s: string) -> i32`. Not in the stdlib yet;
        // this is the minimal way to get a length out of a string without
        // waiting on lumen-l64 / proper method call resolution.
        if args.len() != 1 {
            self.errors.push(TypeError {
                span: call_span,
                message: format!("`string_len` expects 1 argument, found {}", args.len()),
            });
            return Ty::I32;
        }
        self.check_expr(&args[0].value, &Ty::String);
        Ty::I32
    }

    fn check_range_call(&mut self, args: &[Arg], call_span: Span) -> Ty {
        // Built-in stand-in for `std/env.range(start: i32, end: i32)` until
        // the real stdlib lands. Accepts two i32s and yields a List<i32> so
        // `for i in range(0, n) { ... }` typechecks.
        if args.len() != 2 {
            self.errors.push(TypeError {
                span: call_span,
                message: format!("`range` expects 2 arguments, found {}", args.len()),
            });
            return Ty::List(Box::new(Ty::I32));
        }
        self.check_expr(&args[0].value, &Ty::I32);
        self.check_expr(&args[1].value, &Ty::I32);
        Ty::List(Box::new(Ty::I32))
    }

    fn find_variant(&self, name: &str) -> Option<(String, VariantClone)> {
        for (ty_name, info) in &self.module.types {
            if let TypeInfo::Sum { variants, .. } = info {
                for v in variants {
                    if v.name == name {
                        return Some((ty_name.clone(), clone_variant(v)));
                    }
                }
            }
        }
        None
    }

    fn check_variant_call(
        &mut self,
        ty_name: &str,
        variant: VariantClone,
        args: &[Arg],
        call_span: Span,
    ) -> Ty {
        match variant.payload {
            None => {
                if !args.is_empty() {
                    self.errors.push(TypeError {
                        span: call_span,
                        message: format!(
                            "variant `{}` of type `{ty_name}` takes no arguments",
                            variant.name
                        ),
                    });
                }
            }
            Some(VariantPayloadCloneKind::Positional(tys)) => {
                if args.len() != tys.len() {
                    self.errors.push(TypeError {
                        span: call_span,
                        message: format!(
                            "variant `{}` expects {} positional argument(s), found {}",
                            variant.name,
                            tys.len(),
                            args.len()
                        ),
                    });
                } else {
                    for (i, arg) in args.iter().enumerate() {
                        if arg.name.is_some() {
                            self.errors.push(TypeError {
                                span: arg.span,
                                message: format!(
                                    "variant `{}` uses positional fields; drop the name",
                                    variant.name
                                ),
                            });
                        }
                        self.check_expr(&arg.value, &tys[i]);
                    }
                }
            }
            Some(VariantPayloadCloneKind::Named(_)) => {
                // Named-field variants must be constructed with a struct
                // literal `Variant { field: value, ... }`, not as a call.
                self.errors.push(TypeError {
                    span: call_span,
                    message: format!(
                        "variant `{}` has named fields; construct it with `{} {{ ... }}` instead of a call",
                        variant.name, variant.name
                    ),
                });
            }
        }
        Ty::User(ty_name.to_string())
    }

    fn check_field_access(&mut self, recv: &Ty, field: &str, span: Span) -> Ty {
        let name = match recv {
            Ty::User(n) => n.clone(),
            Ty::Error => return Ty::Error,
            other => {
                self.errors.push(TypeError {
                    span,
                    message: format!(
                        "field access requires a struct, found {}",
                        other.display()
                    ),
                });
                return Ty::Error;
            }
        };
        let Some(TypeInfo::Struct { fields, .. }) = self.module.types.get(&name) else {
            self.errors.push(TypeError {
                span,
                message: format!("type `{name}` is not a struct, cannot take a field"),
            });
            return Ty::Error;
        };
        for (fname, fty) in fields {
            if fname == field {
                return fty.clone();
            }
        }
        self.errors.push(TypeError {
            span,
            message: format!("struct `{name}` has no field `{field}`"),
        });
        Ty::Error
    }

    fn check_try(&mut self, inner: &Expr, span: Span) -> Ty {
        let inner_ty = self.infer_expr(inner);
        match (&inner_ty, &self.sig.ret) {
            (Ty::Result(ok, err), Ty::Result(_, ret_err)) => {
                if !compatible(err, ret_err) {
                    self.errors.push(TypeError {
                        span,
                        message: format!(
                            "`?` error type mismatch: inner is {}, function returns {}",
                            err.display(),
                            self.sig.ret.display()
                        ),
                    });
                }
                (**ok).clone()
            }
            (Ty::Option(inner_ok), Ty::Option(_)) => (**inner_ok).clone(),
            (Ty::Error, _) => Ty::Error,
            (other, ret) => {
                self.errors.push(TypeError {
                    span,
                    message: format!(
                        "`?` requires a Result or Option in a function that returns one; here the inner type is {} and the function returns {}",
                        other.display(),
                        ret.display()
                    ),
                });
                Ty::Error
            }
        }
    }

    fn check_match(&mut self, scrutinee: &Expr, arms: &[MatchArm], span: Span) -> Ty {
        let scrut_ty = self.infer_expr(scrutinee);
        let mut arm_ty: Option<Ty> = None;

        for arm in arms {
            self.push_scope();
            self.check_pattern(&arm.pattern, &scrut_ty);
            let body_ty = match &arm_ty {
                Some(t) => self.check_expr(&arm.body, t),
                None => self.infer_expr(&arm.body),
            };
            if arm_ty.is_none() {
                arm_ty = Some(body_ty);
            }
            self.pop_scope();
        }

        match &scrut_ty {
            Ty::User(name) => {
                if let Some(TypeInfo::Sum { variants, .. }) = self.module.types.get(name) {
                    self.check_exhaustiveness(variants, arms, span);
                }
            }
            Ty::Option(_) => {
                let synthetic = vec![
                    VariantInfo {
                        name: "None".into(),
                        payload: None,
                    },
                    VariantInfo {
                        name: "Some".into(),
                        payload: Some(VariantPayloadInfo::Positional(vec![Ty::Error])),
                    },
                ];
                self.check_exhaustiveness(&synthetic, arms, span);
            }
            Ty::Result(_, _) => {
                let synthetic = vec![
                    VariantInfo {
                        name: "Ok".into(),
                        payload: Some(VariantPayloadInfo::Positional(vec![Ty::Error])),
                    },
                    VariantInfo {
                        name: "Err".into(),
                        payload: Some(VariantPayloadInfo::Positional(vec![Ty::Error])),
                    },
                ];
                self.check_exhaustiveness(&synthetic, arms, span);
            }
            _ => {}
        }

        arm_ty.unwrap_or(Ty::Error)
    }

    fn check_pattern(&mut self, pat: &Pattern, expected: &Ty) {
        match &pat.kind {
            PatternKind::Wildcard => {}
            PatternKind::Binding(name) => {
                self.declare(name, expected.clone(), false, pat.span);
            }
            PatternKind::Literal(lit) => {
                let lit_ty = match lit {
                    LiteralPattern::Int(_, suffix) => match suffix {
                        Some(IntSuffix::I32) => Ty::I32,
                        Some(IntSuffix::I64) => Ty::I64,
                        Some(IntSuffix::U32) => Ty::U32,
                        Some(IntSuffix::U64) => Ty::U64,
                        None => Ty::I32,
                    },
                    LiteralPattern::Bool(_) => Ty::Bool,
                    LiteralPattern::String(_) => Ty::String,
                    LiteralPattern::Unit => Ty::Unit,
                };
                if !compatible(&lit_ty, expected) {
                    self.errors.push(TypeError {
                        span: pat.span,
                        message: format!(
                            "literal pattern of type {} does not match scrutinee {}",
                            lit_ty.display(),
                            expected.display()
                        ),
                    });
                }
            }
            PatternKind::Variant { name, payload } => {
                self.check_variant_pattern(name, payload.as_ref(), expected, pat.span);
            }
        }
    }

    fn check_variant_pattern(
        &mut self,
        name: &str,
        payload: Option<&VariantPatPayload>,
        expected: &Ty,
        span: Span,
    ) {
        // Built-in Option / Result patterns.
        match expected {
            Ty::Option(inner) => {
                self.check_builtin_option_pattern(name, payload, inner, span);
                return;
            }
            Ty::Result(ok, err) => {
                self.check_builtin_result_pattern(name, payload, ok, err, span);
                return;
            }
            _ => {}
        }

        let ty_name = match expected {
            Ty::User(n) => n.clone(),
            Ty::Error => return,
            other => {
                self.errors.push(TypeError {
                    span,
                    message: format!(
                        "variant pattern `{name}` cannot match scrutinee of type {}",
                        other.display()
                    ),
                });
                return;
            }
        };
        let Some(TypeInfo::Sum { variants, .. }) = self.module.types.get(&ty_name) else {
            self.errors.push(TypeError {
                span,
                message: format!("type `{ty_name}` is not a sum type"),
            });
            return;
        };
        let Some(variant) = variants.iter().find(|v| v.name == name) else {
            self.errors.push(TypeError {
                span,
                message: format!("type `{ty_name}` has no variant `{name}`"),
            });
            return;
        };

        let variant_clone = clone_variant(variant);

        match (payload, variant_clone.payload) {
            (None, None) => {}
            (Some(VariantPatPayload::Named(sub)), Some(VariantPayloadCloneKind::Named(fields))) => {
                let sub = sub.clone();
                for pf in &sub {
                    let Some((_, fty)) = fields.iter().find(|(fname, _)| fname == &pf.name) else {
                        self.errors.push(TypeError {
                            span: pf.span,
                            message: format!(
                                "variant `{name}` has no field `{}`",
                                pf.name
                            ),
                        });
                        continue;
                    };
                    self.check_pattern(&pf.pattern, fty);
                }
            }
            (
                Some(VariantPatPayload::Positional(subs)),
                Some(VariantPayloadCloneKind::Positional(tys)),
            ) => {
                if subs.len() != tys.len() {
                    self.errors.push(TypeError {
                        span,
                        message: format!(
                            "variant `{name}` expects {} positional subpatterns, found {}",
                            tys.len(),
                            subs.len()
                        ),
                    });
                } else {
                    for (sub, ty) in subs.iter().zip(tys.iter()) {
                        self.check_pattern(sub, ty);
                    }
                }
            }
            (None, Some(_)) => {
                self.errors.push(TypeError {
                    span,
                    message: format!("variant `{name}` has a payload that must be bound"),
                });
            }
            (Some(_), None) => {
                self.errors.push(TypeError {
                    span,
                    message: format!("variant `{name}` has no payload"),
                });
            }
            (Some(VariantPatPayload::Named(_)), Some(VariantPayloadCloneKind::Positional(_))) => {
                self.errors.push(TypeError {
                    span,
                    message: format!(
                        "variant `{name}` has positional fields; use `(` `)` in the pattern"
                    ),
                });
            }
            (Some(VariantPatPayload::Positional(_)), Some(VariantPayloadCloneKind::Named(_))) => {
                self.errors.push(TypeError {
                    span,
                    message: format!(
                        "variant `{name}` has named fields; use `{{` `}}` in the pattern"
                    ),
                });
            }
        }
    }

    fn check_builtin_option_pattern(
        &mut self,
        name: &str,
        payload: Option<&VariantPatPayload>,
        inner: &Ty,
        span: Span,
    ) {
        match (name, payload) {
            ("None", None) => {}
            ("None", Some(_)) => self.errors.push(TypeError {
                span,
                message: "`None` has no payload".into(),
            }),
            ("Some", Some(VariantPatPayload::Positional(pats))) if pats.len() == 1 => {
                self.check_pattern(&pats[0], inner);
            }
            ("Some", _) => self.errors.push(TypeError {
                span,
                message: "`Some(x)` expects one positional sub-pattern".into(),
            }),
            (other, _) => self.errors.push(TypeError {
                span,
                message: format!("`Option` has no variant `{other}` (expected Some or None)"),
            }),
        }
    }

    fn check_builtin_result_pattern(
        &mut self,
        name: &str,
        payload: Option<&VariantPatPayload>,
        ok: &Ty,
        err: &Ty,
        span: Span,
    ) {
        match (name, payload) {
            ("Ok", Some(VariantPatPayload::Positional(pats))) if pats.len() == 1 => {
                self.check_pattern(&pats[0], ok);
            }
            ("Err", Some(VariantPatPayload::Positional(pats))) if pats.len() == 1 => {
                self.check_pattern(&pats[0], err);
            }
            ("Ok" | "Err", _) => self.errors.push(TypeError {
                span,
                message: format!("`{name}(x)` expects one positional sub-pattern"),
            }),
            (other, _) => self.errors.push(TypeError {
                span,
                message: format!("`Result` has no variant `{other}` (expected Ok or Err)"),
            }),
        }
    }

    fn check_exhaustiveness(
        &mut self,
        variants: &[VariantInfo],
        arms: &[MatchArm],
        span: Span,
    ) {
        // A wildcard or a bare binding covers everything.
        let has_wild = arms.iter().any(|a| {
            matches!(
                a.pattern.kind,
                PatternKind::Wildcard | PatternKind::Binding(_)
            )
        });
        if has_wild {
            return;
        }

        let mut covered: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for arm in arms {
            if let PatternKind::Variant { name, .. } = &arm.pattern.kind {
                covered.insert(name.as_str());
            }
        }

        let missing: Vec<String> = variants
            .iter()
            .filter(|v| !covered.contains(v.name.as_str()))
            .map(|v| v.name.clone())
            .collect();
        if !missing.is_empty() {
            self.errors.push(TypeError {
                span,
                message: format!("non-exhaustive match; missing variant(s): {}", missing.join(", ")),
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compatible means "equal *or* one side is Ty::Error (which silences
/// cascades)". Crucially, this is not subtyping — i32 and i64 are NOT
/// compatible; you have to `as`-cast. The one exception is [`Ty::Error`].
fn compatible(a: &Ty, b: &Ty) -> bool {
    match (a, b) {
        (Ty::Error, _) | (_, Ty::Error) => true,
        // Option<Error> from `None` unifies with any Option<T>.
        (Ty::Option(inner), Ty::Option(_)) | (Ty::Option(_), Ty::Option(inner))
            if matches!(**inner, Ty::Error) =>
        {
            true
        }
        // Result<T, Error> from bare Ok(v) / Err(e) unifies with any Result
        // whose T (or E respectively) matches.
        (Ty::Result(oa, ea), Ty::Result(ob, eb)) => {
            (compatible(oa, ob) && compatible(ea, eb))
                || (matches!(**ea, Ty::Error) && compatible(oa, ob))
                || (matches!(**eb, Ty::Error) && compatible(oa, ob))
                || (matches!(**oa, Ty::Error) && compatible(ea, eb))
                || (matches!(**ob, Ty::Error) && compatible(ea, eb))
        }
        (Ty::Tuple(a_elems), Ty::Tuple(b_elems)) => {
            a_elems.len() == b_elems.len()
                && a_elems.iter().zip(b_elems.iter()).all(|(a, b)| compatible(a, b))
        }
        _ => a == b,
    }
}

#[derive(Debug)]
struct VariantClone {
    name: String,
    payload: Option<VariantPayloadCloneKind>,
}

#[derive(Debug)]
enum VariantPayloadCloneKind {
    Named(Vec<(String, Ty)>),
    Positional(Vec<Ty>),
}

fn clone_variant(v: &VariantInfo) -> VariantClone {
    VariantClone {
        name: v.name.clone(),
        payload: v.payload.as_ref().map(|p| match p {
            VariantPayloadInfo::Named(fields) => VariantPayloadCloneKind::Named(fields.clone()),
            VariantPayloadInfo::Positional(tys) => VariantPayloadCloneKind::Positional(tys.clone()),
        }),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse;

    fn tc(src: &str) -> Result<ModuleInfo, Vec<TypeError>> {
        let toks = lex(src).unwrap();
        let m = parse(toks).unwrap();
        typecheck(&m)
    }

    fn tc_ok(src: &str) {
        if let Err(errors) = tc(src) {
            for e in &errors {
                eprintln!("{e}");
            }
            panic!("expected typecheck to succeed");
        }
    }

    fn tc_err(src: &str) -> Vec<TypeError> {
        tc(src).expect_err("expected typecheck to fail")
    }

    // --- Positive cases ---------------------------------------------------

    #[test]
    fn identity_fn() {
        tc_ok("fn id(x: i32): i32 { return x }");
    }

    #[test]
    fn arithmetic_and_comparison() {
        tc_ok("fn f(a: i32, b: i32): bool { return (a + b) * 2 < 100 }");
    }

    #[test]
    fn string_concat_via_plus() {
        tc_ok("fn greet(name: string): string { return \"hi, \" + name }");
    }

    #[test]
    fn as_cast_between_numeric_types() {
        tc_ok("fn widen(n: i32): i64 { return n as i64 }");
    }

    #[test]
    fn var_and_assign_and_for_loop() {
        tc_ok(
            r#"
            fn sum_of_squares(n: i32): i32 {
                var total: i32 = 0
                for i in range(1, n + 1) {
                    total = total + i * i
                }
                return total
            }
            "#,
        );
    }

    #[test]
    fn user_struct_and_field_access() {
        tc_ok(
            r#"
            type User = { name: string, age: i32 }

            fn age_of(u: User): i32 {
                return u.age
            }
            "#,
        );
    }

    #[test]
    fn struct_literal_construction() {
        tc_ok(
            r#"
            type Point = { x: i32, y: i32 }

            fn make(): Point {
                return Point { x: 1, y: 2 }
            }
            "#,
        );
    }

    #[test]
    fn sum_type_exhaustive_match() {
        tc_ok(
            r#"
            type Shape =
                | Circle { radius: f64 }
                | Rectangle { width: f64, height: f64 }
                | Triangle { base: f64, height: f64 }

            fn area(s: Shape): f64 {
                return match s {
                    Circle { radius: r } => r,
                    Rectangle { width: w, height: h } => w,
                    Triangle { base: b, height: h } => b,
                }
            }
            "#,
        );
    }

    #[test]
    fn wildcard_covers_missing_variants() {
        tc_ok(
            r#"
            type T = | A | B | C

            fn f(t: T): i32 {
                return match t {
                    A => 1,
                    _ => 0,
                }
            }
            "#,
        );
    }

    #[test]
    fn result_constructors_and_try() {
        tc_ok(
            r#"
            fn parse(s: string): Result<i32, string> {
                return Ok(42)
            }

            fn use_it(s: string): Result<i32, string> {
                let n = parse(s)?
                return Ok(n)
            }
            "#,
        );
    }

    #[test]
    fn option_constructors_and_try() {
        tc_ok(
            r#"
            fn first(): Option<i32> {
                return Some(1)
            }

            fn second(): Option<i32> {
                let x = first()?
                return Some(x)
            }

            fn third(): Option<i32> {
                return None
            }
            "#,
        );
    }

    #[test]
    fn positional_variant_constructor() {
        tc_ok(
            r#"
            type Token =
                | Number(i32)
                | Word(string)

            fn mk_num(): Token {
                return Number(1)
            }
            "#,
        );
    }

    #[test]
    fn pure_fn_calling_io_fn_is_error() {
        let errs = tc_err(
            r#"
            import std/io
            fn greet(): unit io { io.println("hi") }
            fn bad(): unit { greet() }
            "#,
        );
        assert!(errs.iter().any(|e| e.message.contains("pure") && e.message.contains("io")));
    }

    #[test]
    fn io_fn_calling_io_fn_is_ok() {
        tc_ok(
            r#"
            import std/io
            fn a(): unit io { io.println("a") }
            fn b(): unit io { a() }
            "#,
        );
    }

    #[test]
    fn pure_fn_calling_pure_fn_is_ok() {
        tc_ok("fn a(): i32 { return 1 }\nfn b(): i32 { return a() }");
    }

    #[test]
    fn pure_fn_calling_io_println_directly_is_error() {
        let errs = tc_err(
            r#"
            import std/io
            fn bad(): unit { io.println("oops") }
            "#,
        );
        assert!(errs.iter().any(|e| e.message.contains("pure") && e.message.contains("io")));
    }

    // --- Negative cases: one per rule ------------------------------------

    #[test]
    fn no_implicit_numeric_conversion() {
        let errs = tc_err("fn f(n: i32): i64 { return n }");
        assert!(errs.iter().any(|e| e.message.contains("expected i64")));
    }

    #[test]
    fn mismatched_operands() {
        let errs = tc_err("fn f(a: i32, b: i64): i32 { return a + b }");
        assert!(errs.iter().any(|e| e.message.contains("matching numeric types")));
    }

    #[test]
    fn shadowing_is_an_error() {
        let errs = tc_err(
            r#"
            fn f(): i32 {
                let x: i32 = 1
                let x: i32 = 2
                return x
            }
            "#,
        );
        assert!(errs.iter().any(|e| e.message.contains("shadowing")));
    }

    #[test]
    fn assign_to_let_is_an_error() {
        let errs = tc_err(
            r#"
            fn f(): i32 {
                let x: i32 = 1
                x = 2
                return x
            }
            "#,
        );
        assert!(errs.iter().any(|e| e.message.contains("`let`")));
    }

    #[test]
    fn struct_literal_missing_field() {
        let errs = tc_err(
            r#"
            type Point = { x: i32, y: i32 }

            fn f(): Point {
                return Point { x: 1 }
            }
            "#,
        );
        assert!(errs.iter().any(|e| e.message.contains("missing field `y`")));
    }

    #[test]
    fn non_exhaustive_match_names_missing_variants() {
        let errs = tc_err(
            r#"
            type T = | A | B | C

            fn f(t: T): i32 {
                return match t {
                    A => 1,
                    B => 2,
                }
            }
            "#,
        );
        assert!(errs
            .iter()
            .any(|e| e.message.contains("non-exhaustive") && e.message.contains("C")));
    }

    #[test]
    fn try_in_wrong_function_type() {
        let errs = tc_err(
            r#"
            fn parse(s: string): Result<i32, string> { return Ok(1) }

            fn f(): i32 {
                let n = parse("a")?
                return n
            }
            "#,
        );
        assert!(errs.iter().any(|e| e.message.contains("?")));
    }

    #[test]
    fn cast_requires_numeric() {
        let errs = tc_err("fn f(b: bool): i32 { return b as i32 }");
        assert!(errs.iter().any(|e| e.message.contains("numeric source")));
    }
}
