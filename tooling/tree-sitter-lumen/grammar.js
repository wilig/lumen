/**
 * @file Tree-sitter grammar for Lumen.
 * @author Will Groppe
 * @license MIT
 *
 * Mirrors docs/grammar.ebnf. When the EBNF diverges from the Rust
 * parser (src/parser.rs), the Rust parser wins — fix the EBNF and
 * this grammar in lockstep.
 */

/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

const PREC = {
  or: 1,
  and: 2,
  eq: 3,
  rel: 4,
  add: 5,
  mul: 6,
  cast: 7,
  unary: 8,
  postfix: 9,
  call: 10,
};

module.exports = grammar({
  name: "lumen",

  extras: $ => [
    /\s/,
    $.line_comment,
    $.block_comment,
  ],

  word: $ => $.identifier,

  conflicts: $ => [
    // `IDENT {` could start a struct literal or be a plain identifier
    // followed by a block. The parser resolves via context.
    [$._expression, $.struct_literal],
    // `{ expr }` — the final expression could be a tail or an
    // expr_stmt. Both readings are valid; prec.dynamic on expr_stmt
    // nudges toward the tail interpretation.
    [$.block, $.expr_stmt],
  ],

  rules: {
    source_file: $ => seq(
      repeat($.import),
      repeat($._item),
    ),

    // --- Imports -----------------------------------------------------

    import: $ => seq(
      "import",
      field("path", $.module_path),
      optional(seq("as", field("alias", $.identifier))),
    ),

    module_path: $ => seq(
      $.identifier,
      repeat(seq("/", $.identifier)),
    ),

    // --- Items -------------------------------------------------------

    _item: $ => choice(
      $.fn_decl,
      $.extern_fn_decl,
      $.type_decl,
      $.actor_decl,
      $.msg_handler_decl,
      $.global_let,
    ),

    global_let: $ => seq(
      choice("let", "var"),
      field("name", $.identifier),
      optional(seq(":", field("type", $._type))),
      "=",
      field("value", $._expression),
    ),

    fn_decl: $ => seq(
      "fn",
      field("name", $.identifier),
      optional(field("type_params", $.type_parameters)),
      field("params", $.parameters),
      ":",
      field("return_type", $._type),
      optional(field("effect", $.effect)),
      field("body", $.block),
    ),

    extern_fn_decl: $ => seq(
      "extern",
      "fn",
      field("name", $.identifier),
      optional(field("type_params", $.type_parameters)),
      field("params", $.parameters),
      ":",
      field("return_type", $._type),
      optional(seq("link", field("link_name", $.string_literal))),
    ),

    type_parameters: $ => seq(
      "<",
      $.identifier,
      repeat(seq(",", $.identifier)),
      ">",
    ),

    parameters: $ => seq(
      "(",
      optional(seq(
        $.parameter,
        repeat(seq(",", $.parameter)),
        optional(","),
      )),
      ")",
    ),

    parameter: $ => seq(
      field("name", $.identifier),
      ":",
      field("type", $._type),
    ),

    effect: $ => choice("io", "pure"),

    actor_decl: $ => seq(
      "actor",
      field("name", $.identifier),
      "{",
      optional($._field_list),
      "}",
    ),

    msg_handler_decl: $ => seq(
      "msg",
      field("actor", $.identifier),
      ".",
      field("name", $.identifier),
      "(",
      "self",
      repeat(seq(",", $.parameter)),
      ")",
      ":",
      field("return_type", $._type),
      field("body", $.block),
    ),

    type_decl: $ => seq(
      "type",
      field("name", $.identifier),
      optional(field("type_params", $.type_parameters)),
      "=",
      choice(
        field("body", $.struct_body),
        field("body", $.sum_body),
      ),
    ),

    struct_body: $ => seq(
      "{",
      optional($._field_list),
      "}",
    ),

    _field_list: $ => seq(
      $.field,
      repeat(seq(",", $.field)),
      optional(","),
    ),

    field: $ => seq(
      field("name", $.identifier),
      ":",
      field("type", $._type),
    ),

    sum_body: $ => seq(
      optional("|"),
      $.variant,
      repeat(seq("|", $.variant)),
    ),

    variant: $ => seq(
      field("name", $.identifier),
      optional(field("payload", choice(
        $.variant_named_payload,
        $.variant_positional_payload,
      ))),
    ),

    variant_named_payload: $ => seq(
      "{",
      optional($._field_list),
      "}",
    ),

    variant_positional_payload: $ => seq(
      "(",
      $._type,
      repeat(seq(",", $._type)),
      ")",
    ),

    // --- Types -------------------------------------------------------

    _type: $ => choice(
      $.named_type,
      $.tuple_type,
      $.fn_ptr_type,
    ),

    // `IDENT<...>` always means a generic instantiation in type
    // position — eager right-associative choice so `as IDENT<...>`
    // is read as cast-to-generic, not cast-to-IDENT then compare.
    named_type: $ => prec.right(20, seq(
      field("name", $.identifier),
      optional($.type_arguments),
    )),

    type_arguments: $ => seq(
      "<",
      $._type,
      repeat(seq(",", $._type)),
      ">",
    ),

    tuple_type: $ => seq(
      "(",
      $._type,
      repeat1(seq(",", $._type)),
      optional(","),
      ")",
    ),

    fn_ptr_type: $ => seq(
      "fn",
      "(",
      optional(seq(
        $._type,
        repeat(seq(",", $._type)),
      )),
      ")",
      ":",
      $._type,
    ),

    // --- Statements --------------------------------------------------

    block: $ => seq(
      "{",
      repeat($._stmt),
      optional(field("tail", $._expression)),
      "}",
    ),

    _stmt: $ => choice(
      $.let_stmt,
      $.let_tuple_stmt,
      $.var_stmt,
      $.assign_stmt,
      $.for_stmt,
      $.return_stmt,
      $.expr_stmt,
    ),

    let_stmt: $ => seq(
      "let",
      field("name", $.identifier),
      optional(seq(":", field("type", $._type))),
      "=",
      field("value", $._expression),
    ),

    let_tuple_stmt: $ => seq(
      "let",
      "(",
      $.identifier,
      repeat1(seq(",", $.identifier)),
      ")",
      "=",
      field("value", $._expression),
    ),

    var_stmt: $ => seq(
      "var",
      field("name", $.identifier),
      optional(seq(":", field("type", $._type))),
      "=",
      field("value", $._expression),
    ),

    assign_stmt: $ => prec(-1, seq(
      field("target", $.identifier),
      field("op", choice("=", "+=", "-=", "*=")),
      field("value", $._expression),
    )),

    for_stmt: $ => seq(
      "for",
      field("binder", $.identifier),
      "in",
      field("iter", $._expression),
      field("body", $.block),
    ),

    return_stmt: $ => prec.right(seq(
      "return",
      optional(field("value", $._expression)),
    )),

    // Lower dynamic prec than block's tail expression so `{ ... expr }`
    // parses `expr` as the tail rather than a final expr_stmt.
    expr_stmt: $ => prec.dynamic(-1, $._expression),

    // --- Expressions -------------------------------------------------

    _expression: $ => choice(
      $.if_expr,
      $.match_expr,
      $.binary_expr,
      $.unary_expr,
      $.cast_expr,
      $.call_expr,
      $.method_call_expr,
      $.field_expr,
      $.tuple_field_expr,
      $.try_expr,
      $.block,
      $.paren_expr,
      $.tuple_literal,
      $.struct_literal,
      $.lambda,
      $.spawn_expr,
      $.send_expr,
      $.ask_expr,
      $.arena_expr,
      $.interpolated_string,
      $.identifier,
      $._literal,
    ),

    if_expr: $ => prec.right(seq(
      "if",
      field("cond", $._expression),
      field("then", $.block),
      optional(seq(
        "else",
        field("else", choice($.if_expr, $.block)),
      )),
    )),

    match_expr: $ => seq(
      "match",
      field("scrutinee", $._expression),
      "{",
      optional(seq(
        $.match_arm,
        repeat(seq(",", $.match_arm)),
        optional(","),
      )),
      "}",
    ),

    match_arm: $ => seq(
      field("pattern", $._pattern),
      "=>",
      field("body", $._expression),
    ),

    binary_expr: $ => {
      const table = [
        ["||", PREC.or, "left"],
        ["&&", PREC.and, "left"],
        ["==", PREC.eq, "left"],
        ["!=", PREC.eq, "left"],
        ["<",  PREC.rel, "left"],
        ["<=", PREC.rel, "left"],
        [">",  PREC.rel, "left"],
        [">=", PREC.rel, "left"],
        ["+",  PREC.add, "left"],
        ["-",  PREC.add, "left"],
        ["*",  PREC.mul, "left"],
        ["/",  PREC.mul, "left"],
        ["%",  PREC.mul, "left"],
      ];
      return choice(...table.map(([op, p, _assoc]) =>
        prec.left(p, seq(
          field("left", $._expression),
          field("op", op),
          field("right", $._expression),
        )),
      ));
    },

    unary_expr: $ => prec(PREC.unary, seq(
      field("op", choice("-", "!")),
      field("operand", $._expression),
    )),

    cast_expr: $ => prec.left(PREC.cast, seq(
      field("expr", $._expression),
      "as",
      field("type", $._type),
    )),

    call_expr: $ => prec(PREC.call, seq(
      field("callee", $._expression),
      field("args", $.call_arguments),
    )),

    method_call_expr: $ => prec(PREC.call, seq(
      field("receiver", $._expression),
      ".",
      field("method", $.identifier),
      field("args", $.call_arguments),
    )),

    field_expr: $ => prec(PREC.postfix, seq(
      field("receiver", $._expression),
      ".",
      field("field", $.identifier),
    )),

    tuple_field_expr: $ => prec(PREC.postfix, seq(
      field("receiver", $._expression),
      ".",
      field("index", $.integer_literal),
    )),

    try_expr: $ => prec(PREC.postfix, seq(
      field("expr", $._expression),
      "?",
    )),

    call_arguments: $ => seq(
      "(",
      optional(seq(
        $.argument,
        repeat(seq(",", $.argument)),
        optional(","),
      )),
      ")",
    ),

    argument: $ => seq(
      optional(seq(field("name", $.identifier), ":")),
      field("value", $._expression),
    ),

    paren_expr: $ => seq("(", $._expression, ")"),

    tuple_literal: $ => prec(1, seq(
      "(",
      $._expression,
      ",",
      optional(seq(
        $._expression,
        repeat(seq(",", $._expression)),
        optional(","),
      )),
      ")",
    )),

    struct_literal: $ => seq(
      field("name", $.identifier),
      "{",
      optional($._field_init_list),
      "}",
    ),

    _field_init_list: $ => seq(
      choice($.field_init, $.spread),
      repeat(seq(",", choice($.field_init, $.spread))),
      optional(","),
    ),

    field_init: $ => seq(
      field("name", $.identifier),
      ":",
      field("value", $._expression),
    ),

    spread: $ => seq("..", field("source", $._expression)),

    lambda: $ => seq(
      "fn",
      field("params", $.parameters),
      ":",
      field("return_type", $._type),
      field("body", $.block),
    ),

    spawn_expr: $ => seq(
      "spawn",
      field("actor", $.identifier),
      "{",
      optional($._field_init_list),
      "}",
    ),

    send_expr: $ => prec(PREC.call + 1, seq(
      "send",
      field("handle", $._expression),
      ".",
      field("method", $.identifier),
      field("args", $.call_arguments),
    )),

    ask_expr: $ => prec(PREC.call + 1, seq(
      "ask",
      field("handle", $._expression),
      ".",
      field("method", $.identifier),
      field("args", $.call_arguments),
    )),

    arena_expr: $ => seq(
      "arena",
      field("body", $.block),
    ),

    // --- Patterns ----------------------------------------------------

    _pattern: $ => choice(
      $.wildcard_pattern,
      $.literal_pattern,
      $.variant_pattern,
      $.binding_pattern,
    ),

    wildcard_pattern: $ => "_",

    literal_pattern: $ => choice(
      $.integer_literal,
      $.float_literal,
      $.string_literal,
      $.char_literal,
      $.boolean_literal,
      $.unit_literal,
    ),

    binding_pattern: $ => $.identifier,

    // A bare identifier is parsed as binding_pattern; variant_pattern
    // requires a payload so `Some(n)` binds `n` rather than treating
    // it as a zero-arg variant. The typechecker later reinterprets
    // bare-identifier patterns that match a known variant (e.g.
    // `None`) as variants — mirrors the Rust parser's behavior.
    variant_pattern: $ => seq(
      field("name", $.identifier),
      field("payload", choice(
        $.variant_pattern_named,
        $.variant_pattern_positional,
      )),
    ),

    variant_pattern_named: $ => seq(
      "{",
      optional(seq(
        $.pattern_field,
        repeat(seq(",", $.pattern_field)),
        optional(","),
      )),
      "}",
    ),

    variant_pattern_positional: $ => seq(
      "(",
      $._pattern,
      repeat(seq(",", $._pattern)),
      ")",
    ),

    pattern_field: $ => seq(
      field("name", $.identifier),
      ":",
      field("pattern", $._pattern),
    ),

    // --- Literals ----------------------------------------------------

    _literal: $ => choice(
      $.integer_literal,
      $.float_literal,
      $.string_literal,
      $.char_literal,
      $.byte_literal,
      $.boolean_literal,
      $.unit_literal,
    ),

    integer_literal: $ => token(seq(
      choice(
        /0x[0-9a-fA-F_]+/,
        /0b[01_]+/,
        /[0-9][0-9_]*/,
      ),
      optional(choice("i32", "i64", "u32", "u64")),
    )),

    float_literal: $ => token(choice(
      // With decimal point
      /[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9]+)?/,
      // Exponent only
      /[0-9][0-9_]*[eE][+-]?[0-9]+/,
    )),

    // Simple (non-interpolated) string literal. Interpolated strings
    // use $.interpolated_string (see below).
    string_literal: $ => token(seq(
      '"',
      repeat(choice(
        /[^"\\$]/,
        /\\./,
        /\$[^{]/,
      )),
      '"',
    )),

    // `"hello ${expr}!"` — literal chunks and `${...}` embeds. Modeled
    // as an explicit node so highlighters can mark the embedded expr.
    interpolated_string: $ => seq(
      '"',
      repeat(choice(
        $.string_content,
        $.string_interpolation,
        $.escape_sequence,
      )),
      '"',
    ),

    string_content: $ => token.immediate(/[^"\\$]+/),
    escape_sequence: $ => token.immediate(/\\./),

    string_interpolation: $ => seq(
      token.immediate("${"),
      $._expression,
      "}",
    ),

    char_literal: $ => token(seq(
      "'",
      choice(
        /\\u\{[0-9a-fA-F]+\}/,
        /\\[nrt\\'"0]/,
        /\\x[0-9a-fA-F]{2}/,
        /[^'\\]/,
      ),
      "'",
    )),

    byte_literal: $ => token(seq(
      "b'",
      choice(
        /\\[nrt\\'"0]/,
        /\\x[0-9a-fA-F]{2}/,
        /[^'\\]/,
      ),
      "'",
    )),

    boolean_literal: $ => choice("true", "false"),
    unit_literal: $ => "unit",

    identifier: $ => /[a-zA-Z_][a-zA-Z_0-9]*/,

    // --- Comments ----------------------------------------------------

    line_comment: $ => token(seq("//", /.*/)),

    // Nested block comments (per grammar.ebnf).
    block_comment: $ => token(seq(
      "/*",
      /([^*]|\*[^/])*/,
      "*/",
    )),
  },
});
