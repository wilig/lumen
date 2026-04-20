; Tree-sitter highlight queries for Lumen.
; Uses the standard capture names understood by nvim-treesitter,
; Helix, Zed, and Emacs treesit.

; --- Keywords -----------------------------------------------------------

[
  "fn"
  "extern"
  "type"
  "actor"
  "msg"
  "import"
  "as"
  "link"
  "let"
  "var"
  "return"
  "for"
  "in"
  "if"
  "else"
  "match"
  "spawn"
  "send"
  "ask"
  "arena"
] @keyword

[
  "io"
  "pure"
] @keyword.modifier

(boolean_literal) @boolean
(unit_literal) @constant.builtin
(wildcard_pattern) @constant.builtin

; --- Operators ----------------------------------------------------------

[
  "=" "+=" "-=" "*="
  "==" "!=" "<" "<=" ">" ">="
  "&&" "||" "!"
  "+" "-" "*" "/" "%"
  "?"
  "=>"
  "|"
  ".."
] @operator

; --- Punctuation --------------------------------------------------------

[ "(" ")" "{" "}" ] @punctuation.bracket
[ "," ":" "." "/" ] @punctuation.delimiter

; --- Types --------------------------------------------------------------

; Built-in type names.
((identifier) @type.builtin
  (#match? @type.builtin "^(i32|i64|u32|u64|f64|bool|string|bytes|unit|char)$"))

; Generic built-ins (Option/Result/List/Map/Handle).
(named_type
  name: (identifier) @type.builtin
  (#match? @type.builtin "^(Option|Result|List|Map|Handle)$"))

(named_type
  name: (identifier) @type)

(type_parameters (identifier) @type.parameter)

; --- Declarations -------------------------------------------------------

(fn_decl name: (identifier) @function)
(extern_fn_decl name: (identifier) @function)
(msg_handler_decl name: (identifier) @function.method)
(type_decl name: (identifier) @type.definition)
(actor_decl name: (identifier) @type.definition)
(variant name: (identifier) @constructor)

(parameter name: (identifier) @variable.parameter)
(field name: (identifier) @property)
(field_init name: (identifier) @property)
(pattern_field name: (identifier) @property)

; --- Calls --------------------------------------------------------------

(call_expr
  callee: (identifier) @function.call)

(method_call_expr
  method: (identifier) @function.method.call)

(send_expr method: (identifier) @function.method.call)
(ask_expr method: (identifier) @function.method.call)
(spawn_expr actor: (identifier) @type)

; Constructors heuristic: CamelCase identifier in call position.
((call_expr
   callee: (identifier) @constructor)
 (#match? @constructor "^[A-Z]"))

; Variant patterns (Some, None, Ok, Err, user variants).
(variant_pattern name: (identifier) @constructor)

; --- Literals -----------------------------------------------------------

(integer_literal) @number
(float_literal) @number.float
(char_literal) @character
(byte_literal) @character
(string_literal) @string
(string_content) @string
(escape_sequence) @string.escape

(string_interpolation
  "${" @punctuation.special
  "}" @punctuation.special)

; Interpolation delimiters inside interpolated strings.

; --- Comments -----------------------------------------------------------

(line_comment) @comment
(block_comment) @comment

; --- Variables ----------------------------------------------------------

(let_stmt name: (identifier) @variable)
(var_stmt name: (identifier) @variable)
(global_let name: (identifier) @variable)
(for_stmt binder: (identifier) @variable)
(binding_pattern (identifier) @variable)

; Fallback for bare identifiers.
(identifier) @variable
