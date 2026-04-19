//! Lumen AST definitions. See `lumen-2s7` and `docs/grammar.ebnf`.
//!
//! Every node carries a [`Span`] so the type checker and codegen can emit
//! source-located errors and the error-frame runtime (`lumen-x4a`) can
//! attribute `?` sites to their originating line/col.

pub use crate::span::Span;
use crate::lexer::IntSuffix;

// ---------------------------------------------------------------------------
// Top level
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Module {
    pub imports: Vec<Import>,
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub struct Import {
    /// Path components: `import std/io` → `["std", "io"]`.
    pub path: Vec<String>,
    /// Optional alias: `import std/raylib as rl` → Some("rl").
    pub alias: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Item {
    Fn(FnDecl),
    Type(TypeDecl),
    ExternFn(ExternFnDecl),
    Actor(ActorDecl),
    MsgHandler(MsgHandlerDecl),
}

/// `actor Counter { count: i32 }`
#[derive(Debug, Clone)]
pub struct ActorDecl {
    pub name: String,
    pub name_span: Span,
    pub fields: Vec<Field>,
    pub span: Span,
}

/// `msg Counter.increment(self, n: i32): Counter { ... }`
#[derive(Debug, Clone)]
pub struct MsgHandlerDecl {
    pub actor_name: String,
    pub name: String,
    pub name_span: Span,
    /// Params excluding `self` (self is implicit).
    pub params: Vec<Param>,
    pub return_type: Type,
    pub body: Block,
    pub span: Span,
}

/// `extern fn malloc(size: i64): i64`
/// `extern fn sqrt(x: f64): f64 link "lumen_sqrt"`
/// Declares an external C-ABI function resolved by the linker.
/// If `link_name` is set, that symbol is used for linking instead of `name`.
#[derive(Debug, Clone)]
pub struct ExternFnDecl {
    pub name: String,
    pub name_span: Span,
    pub params: Vec<Param>,
    pub return_type: Type,
    /// Optional C symbol name: `link "lumen_sqrt"`.
    pub link_name: Option<String>,
    pub span: Span,
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FnDecl {
    pub name: String,
    pub name_span: Span,
    pub params: Vec<Param>,
    pub return_type: Type,
    pub effect: Effect,
    pub body: Block,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: Type,
    pub span: Span,
}

/// Effect annotation on a function. `pure` is the default when the source
/// omits an annotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    Pure,
    Io,
}

// ---------------------------------------------------------------------------
// Type declarations
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TypeDecl {
    pub name: String,
    pub name_span: Span,
    pub body: TypeBody,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum TypeBody {
    Struct(Vec<Field>),
    Sum(Vec<Variant>),
}

#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Variant {
    pub name: String,
    pub payload: Option<VariantPayload>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum VariantPayload {
    /// `{ x: T, y: U }` — named fields.
    Named(Vec<Field>),
    /// `( T, U )` — positional types.
    Positional(Vec<Type>),
}

// ---------------------------------------------------------------------------
// Types (in the type expression sense)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Type {
    pub kind: TypeKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum TypeKind {
    /// A named type, possibly with type arguments.
    Named {
        name: String,
        args: Vec<Type>,
    },
    /// `(T1, T2, ...)` — tuple type.
    Tuple(Vec<Type>),
    /// `fn(T1, T2): R` — function pointer type.
    FnPtr {
        params: Vec<Type>,
        ret: Box<Type>,
    },
}

// ---------------------------------------------------------------------------
// Statements and blocks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    /// Optional final expression whose value is the value of the block.
    pub tail: Option<Box<Expr>>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum StmtKind {
    /// Immutable binding.
    Let {
        name: String,
        ty: Option<Type>,
        value: Expr,
    },
    /// Mutable binding — can be re-assigned via [`StmtKind::Assign`].
    Var {
        name: String,
        ty: Option<Type>,
        value: Expr,
    },
    /// Re-assignment of an existing `var` binding.
    Assign {
        name: String,
        value: Expr,
    },
    /// `let (a, b) = expr` — destructuring a tuple into multiple bindings.
    LetTuple {
        names: Vec<String>,
        value: Expr,
    },
    Expr(Expr),
    For {
        binder: String,
        iter: Expr,
        body: Block,
    },
    Return(Option<Expr>),
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    // Literals
    IntLit {
        value: u64,
        suffix: Option<IntSuffix>,
    },
    FloatLit(f64),
    StringLit(String),
    BoolLit(bool),
    UnitLit,

    // Primary
    Ident(String),
    Paren(Box<Expr>),
    Block(Box<Block>),
    StructLit {
        name: String,
        name_span: Span,
        fields: Vec<FieldInit>,
        /// `..expr` spread: fill unspecified fields from this expression.
        spread: Option<Box<Expr>>,
    },

    // Postfix
    Call {
        callee: Box<Expr>,
        args: Vec<Arg>,
    },
    Field {
        receiver: Box<Expr>,
        name: String,
    },
    MethodCall {
        receiver: Box<Expr>,
        method: String,
        args: Vec<Arg>,
    },
    /// `expr?` — error propagation.
    Try(Box<Expr>),

    /// `(expr, expr, ...)` — tuple literal.
    TupleLit(Vec<Expr>),
    /// `expr.0`, `expr.1` — tuple field access by index.
    TupleField {
        receiver: Box<Expr>,
        index: u32,
    },

    /// `spawn Counter { count: 0 }`
    Spawn {
        actor_name: String,
        fields: Vec<FieldInit>,
    },
    /// `send handle.method(args)`
    Send {
        handle: Box<Expr>,
        method: String,
        args: Vec<Arg>,
    },
    /// `ask handle.method(args)`
    Ask {
        handle: Box<Expr>,
        method: String,
        args: Vec<Arg>,
    },

    // Unary / binary / cast
    Unary {
        op: UnaryOp,
        rhs: Box<Expr>,
    },
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Cast {
        expr: Box<Expr>,
        to: Type,
    },

    // Control flow
    If {
        cond: Box<Expr>,
        then_block: Block,
        else_block: Block,
    },
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },

    /// Non-capturing lambda: `fn(x: i32, y: i32): i32 { return x + y }`
    Lambda {
        params: Vec<Param>,
        return_type: Type,
        body: Block,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    NotEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Debug, Clone)]
pub struct FieldInit {
    pub name: String,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Arg {
    /// Optional named-argument label: `foo(x: 1, y: 2)`.
    pub name: Option<String>,
    pub value: Expr,
    pub span: Span,
}

// ---------------------------------------------------------------------------
// Patterns (for match)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Expr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Pattern {
    pub kind: PatternKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum PatternKind {
    Wildcard,
    Literal(LiteralPattern),
    /// A bare identifier that binds the scrutinee value.
    Binding(String),
    /// A variant constructor pattern. Distinguished from `Binding` by the
    /// parser when a payload is present; the typechecker disambiguates the
    /// no-payload case (where a bare `None` vs a bare `x` can only be
    /// decided once we know what's in scope).
    Variant {
        name: String,
        payload: Option<VariantPatPayload>,
    },
}

#[derive(Debug, Clone)]
pub enum LiteralPattern {
    Int(u64, Option<IntSuffix>),
    Bool(bool),
    String(String),
    Unit,
}

#[derive(Debug, Clone)]
pub enum VariantPatPayload {
    Named(Vec<PatField>),
    Positional(Vec<Pattern>),
}

#[derive(Debug, Clone)]
pub struct PatField {
    pub name: String,
    pub pattern: Pattern,
    pub span: Span,
}
