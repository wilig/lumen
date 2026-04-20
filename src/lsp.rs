//! Language Server Protocol (lumen-sa7).
//!
//! Speaks JSON-RPC 2.0 over stdin/stdout to any LSP-capable editor
//! (nvim-lspconfig, vscode-languageclient, Helix, Zed, Emacs
//! eglot). The server re-runs the existing lex/parse/typecheck
//! pipeline on every document change and publishes diagnostics.
//!
//! Scope today:
//!   - diagnostics (parse + type errors, with file:line:col + range)
//!   - hover (show the type of the symbol under the cursor when we
//!     can resolve it from the most recent typecheck)
//!
//! Out of scope (follow-ups): go-to-definition, find-references,
//! rename, autocomplete, workspace-wide analysis.

use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::ast;
use crate::imports;
use crate::lexer;
use crate::parser;
use crate::span::Span;
use crate::types::{self, FnSig, ModuleInfo, Ty, TypeError};

/// Run the LSP server on stdin/stdout until a shutdown+exit sequence
/// arrives. Returns the process exit code the CLI should use.
pub fn run() -> i32 {
    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut server = Server::default();

    loop {
        let msg = match read_message(&mut reader) {
            Ok(Some(v)) => v,
            Ok(None) => return 0, // EOF
            Err(e) => {
                eprintln!("lumen-lsp: read error: {e}");
                return 1;
            }
        };
        let responses = server.handle(msg);
        for resp in responses {
            if write_message(&mut out, &resp).is_err() {
                return 1;
            }
        }
        if server.shutting_down && server.exit_requested {
            return 0;
        }
    }
}

// --- Message framing ----------------------------------------------------

fn read_message(reader: &mut impl BufRead) -> io::Result<Option<Value>> {
    let mut content_length: Option<usize> = None;
    let mut header_line = String::new();
    loop {
        header_line.clear();
        let n = reader.read_line(&mut header_line)?;
        if n == 0 {
            return Ok(None); // EOF
        }
        let line = header_line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            break; // blank line → end of headers
        }
        if let Some(rest) = line.strip_prefix("Content-Length:") {
            content_length = rest.trim().parse().ok();
        }
        // Ignore other headers (Content-Type).
    }
    let len = content_length.ok_or_else(|| io::Error::new(
        io::ErrorKind::InvalidData,
        "missing Content-Length",
    ))?;
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body)?;
    let v = serde_json::from_slice::<Value>(&body)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(Some(v))
}

fn write_message(out: &mut impl Write, v: &Value) -> io::Result<()> {
    let body = serde_json::to_vec(v)?;
    write!(out, "Content-Length: {}\r\n\r\n", body.len())?;
    out.write_all(&body)?;
    out.flush()
}

// --- Server state + dispatch --------------------------------------------

#[derive(Default)]
struct Server {
    /// Per-document state — source + cached parse/typecheck results
    /// so hover/go-to-def requests don't re-run the whole pipeline.
    docs: HashMap<String, DocState>,
    /// Workspace root discovered via the client's `rootUri` at
    /// initialize time, falling back to the compiler source tree
    /// (`CARGO_MANIFEST_DIR`) so the built-in `std/` modules always
    /// resolve. Imports are loaded relative to this directory.
    workspace_root: Option<PathBuf>,
    shutting_down: bool,
    exit_requested: bool,
}

#[derive(Default)]
struct DocState {
    text: String,
    /// Last successful parse of `text`. If the current text fails to
    /// parse, we keep the previous parse around so hover keeps
    /// working while the user types.
    last_ast: Option<ast::Module>,
    /// Last successful typecheck. Same fallback semantics as last_ast.
    last_info: Option<ModuleInfo>,
}

impl Server {
    fn handle(&mut self, msg: Value) -> Vec<Value> {
        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let id = msg.get("id").cloned();
        let params = msg.get("params").cloned().unwrap_or(Value::Null);

        match method {
            "initialize" => {
                // Extract the workspace root from rootUri /
                // workspaceFolders. Fall back to the compile tree's
                // directory so the built-in std/* modules always
                // resolve even for one-off files outside a project.
                self.workspace_root = extract_workspace_root(&params);
                vec![respond(id, initialize_result())]
            }
            "initialized" => Vec::new(),
            "shutdown" => {
                self.shutting_down = true;
                vec![respond(id, Value::Null)]
            }
            "exit" => {
                self.exit_requested = true;
                Vec::new()
            }
            "textDocument/didOpen" => self.on_did_open(&params),
            "textDocument/didChange" => self.on_did_change(&params),
            "textDocument/didClose" => {
                if let Some(uri) = text_document_uri(&params) {
                    self.docs.remove(uri);
                }
                Vec::new()
            }
            "textDocument/didSave" => self.on_did_save(&params),
            "textDocument/hover" => {
                let result = self.on_hover(&params);
                vec![respond(id, result)]
            }
            "textDocument/definition" => {
                let result = self.on_definition(&params);
                vec![respond(id, result)]
            }
            "textDocument/references" => {
                let result = self.on_references(&params);
                vec![respond(id, result)]
            }
            "textDocument/rename" => {
                let result = self.on_rename(&params);
                vec![respond(id, result)]
            }
            // Any request we don't handle gets a null result so the
            // client doesn't time out. Notifications we don't handle
            // are silently dropped.
            _ if id.is_some() => vec![respond(id, Value::Null)],
            _ => Vec::new(),
        }
    }

    fn on_did_open(&mut self, params: &Value) -> Vec<Value> {
        let Some(uri) = params.pointer("/textDocument/uri").and_then(|u| u.as_str()) else {
            return Vec::new();
        };
        let text = params.pointer("/textDocument/text")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();
        let uri = uri.to_string();
        self.docs.insert(uri.clone(), DocState { text, ..Default::default() });
        vec![self.refresh(&uri)]
    }

    fn on_did_change(&mut self, params: &Value) -> Vec<Value> {
        let Some(uri) = text_document_uri(params) else { return Vec::new(); };
        let uri = uri.to_string();
        // We advertise full-text sync, so each change carries the
        // complete document. Partial-sync support is a follow-up.
        if let Some(changes) = params.pointer("/contentChanges").and_then(|c| c.as_array()) {
            if let Some(last) = changes.last() {
                if let Some(text) = last.get("text").and_then(|t| t.as_str()) {
                    let entry = self.docs.entry(uri.clone()).or_default();
                    entry.text = text.to_string();
                }
            }
        }
        vec![self.refresh(&uri)]
    }

    fn on_did_save(&mut self, params: &Value) -> Vec<Value> {
        let Some(uri) = text_document_uri(params) else { return Vec::new(); };
        let uri = uri.to_string();
        vec![self.refresh(&uri)]
    }

    fn on_hover(&self, params: &Value) -> Value {
        let Some(uri) = text_document_uri(params) else { return Value::Null; };
        let Some(doc) = self.docs.get(uri) else { return Value::Null; };
        let line = params.pointer("/position/line").and_then(|l| l.as_u64()).unwrap_or(0) as u32;
        let col = params.pointer("/position/character").and_then(|c| c.as_u64()).unwrap_or(0) as u32;
        let word = word_at_position(&doc.text, line, col);
        if word.is_empty() { return Value::Null; }
        let markdown = resolve_hover(&word, doc)
            .unwrap_or_else(|| format!("`{word}`"));
        json!({
            "contents": { "kind": "markdown", "value": markdown }
        })
    }

    fn on_definition(&self, params: &Value) -> Value {
        let Some(uri) = text_document_uri(params) else { return Value::Null; };
        let Some(doc) = self.docs.get(uri) else { return Value::Null; };
        let (line, col) = cursor(params);
        let word = word_at_position(&doc.text, line, col);
        if word.is_empty() { return Value::Null; }
        let Some(module) = &doc.last_ast else { return Value::Null; };
        let Some(span) = find_declaration_span(module, &word) else { return Value::Null; };
        let range = span_to_range(&doc.text, &span, word.len() as u32);
        json!({ "uri": uri, "range": range })
    }

    fn on_references(&self, params: &Value) -> Value {
        let Some(uri) = text_document_uri(params) else { return Value::Null; };
        let Some(doc) = self.docs.get(uri) else { return Value::Null; };
        let (line, col) = cursor(params);
        let word = word_at_position(&doc.text, line, col);
        if word.is_empty() { return json!([]); }
        let include_decl = params.pointer("/context/includeDeclaration")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let locations = find_all_occurrences(&doc.text, &word, include_decl);
        let result: Vec<Value> = locations.into_iter()
            .map(|r| json!({ "uri": uri, "range": r }))
            .collect();
        Value::Array(result)
    }

    fn on_rename(&self, params: &Value) -> Value {
        let Some(uri) = text_document_uri(params) else { return Value::Null; };
        let Some(doc) = self.docs.get(uri) else { return Value::Null; };
        let (line, col) = cursor(params);
        let word = word_at_position(&doc.text, line, col);
        if word.is_empty() { return Value::Null; }
        let new_name = match params.pointer("/newName").and_then(|v| v.as_str()) {
            Some(n) if is_valid_identifier(n) => n,
            _ => return Value::Null,
        };
        let occurrences = find_all_occurrences(&doc.text, &word, true);
        let edits: Vec<Value> = occurrences.into_iter()
            .map(|range| json!({ "range": range, "newText": new_name }))
            .collect();
        json!({ "changes": { uri: edits } })
    }

    /// Re-run lex / parse / typecheck for the document, refresh the
    /// cached AST + ModuleInfo (so hover sees current state), and
    /// return a publishDiagnostics notification with any errors.
    fn refresh(&mut self, uri: &str) -> Value {
        let src = self.docs.get(uri).map(|d| d.text.clone()).unwrap_or_default();
        let root = self.resolve_root();
        let (diags, new_ast, new_info) = run_pipeline(&src, &root);
        if let Some(doc) = self.docs.get_mut(uri) {
            if let Some(ast) = new_ast { doc.last_ast = Some(ast); }
            if let Some(info) = new_info { doc.last_info = Some(info); }
        }
        json!({
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": uri,
                "diagnostics": diags,
            }
        })
    }

    fn resolve_root(&self) -> PathBuf {
        self.workspace_root
            .clone()
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
    }
}

/// Parse the workspace root out of LSP `initialize` params. Prefers
/// `workspaceFolders[0].uri`, then `rootUri`, then `rootPath`.
fn extract_workspace_root(params: &Value) -> Option<PathBuf> {
    if let Some(folders) = params.pointer("/workspaceFolders").and_then(|v| v.as_array()) {
        if let Some(uri) = folders.first().and_then(|f| f.get("uri")).and_then(|u| u.as_str()) {
            if let Some(p) = uri_to_path(uri) { return Some(p); }
        }
    }
    if let Some(uri) = params.pointer("/rootUri").and_then(|u| u.as_str()) {
        if let Some(p) = uri_to_path(uri) { return Some(p); }
    }
    if let Some(path) = params.pointer("/rootPath").and_then(|u| u.as_str()) {
        return Some(PathBuf::from(path));
    }
    None
}

fn uri_to_path(uri: &str) -> Option<PathBuf> {
    // file:///home/foo → /home/foo. Only the file:// scheme is useful.
    let rest = uri.strip_prefix("file://")?;
    // Some clients include a host (file://host/path); most use file:///path.
    let path = if rest.starts_with('/') { rest.to_string() } else {
        // file://host/path — drop host.
        match rest.find('/') {
            Some(i) => rest[i..].to_string(),
            None => return None,
        }
    };
    Some(PathBuf::from(path))
}

fn text_document_uri(params: &Value) -> Option<&str> {
    params.pointer("/textDocument/uri").and_then(|u| u.as_str())
}

fn cursor(params: &Value) -> (u32, u32) {
    let line = params.pointer("/position/line").and_then(|l| l.as_u64()).unwrap_or(0) as u32;
    let col = params.pointer("/position/character").and_then(|c| c.as_u64()).unwrap_or(0) as u32;
    (line, col)
}

/// Find the declaration span for `word` among top-level fn/type
/// decls and (not yet) scoped params/locals. Returns the Span of the
/// identifier in the source.
fn find_declaration_span(module: &ast::Module, word: &str) -> Option<Span> {
    use ast::Item;
    for item in &module.items {
        match item {
            Item::Fn(f) if f.name == word => return Some(f.name_span),
            Item::ExternFn(ef) if ef.name == word => return Some(ef.name_span),
            Item::Type(td) if td.name == word => return Some(td.name_span),
            Item::Actor(a) if a.name == word => return Some(a.name_span),
            Item::GlobalLet(g) if g.name == word => return Some(g.name_span),
            _ => {}
        }
    }
    // Walk fn bodies for a matching param or let/var.
    for item in &module.items {
        if let Item::Fn(f) = item {
            for p in &f.params {
                if p.name == word {
                    return Some(p.span);
                }
            }
            if let Some(span) = find_binding_span_in_block(&f.body, word) {
                return Some(span);
            }
        }
    }
    None
}

fn find_binding_span_in_block(block: &ast::Block, word: &str) -> Option<Span> {
    use ast::StmtKind;
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::Let { name, .. } | StmtKind::Var { name, .. } if name == word => {
                return Some(stmt.span);
            }
            StmtKind::For { binder, body, .. } => {
                if binder == word { return Some(stmt.span); }
                if let Some(s) = find_binding_span_in_block(body, word) { return Some(s); }
            }
            _ => {}
        }
    }
    None
}

/// Scan the source text for every whole-word occurrence of `word`.
/// This is a textual match — it doesn't distinguish scopes, but
/// combined with the typechecker's guarantee that top-level names
/// are unique it's good enough for the common refactor cases.
/// Returns LSP ranges (0-based line + character).
fn find_all_occurrences(src: &str, word: &str, _include_decl: bool) -> Vec<Value> {
    let mut out = Vec::new();
    for (line_idx, line) in src.lines().enumerate() {
        let bytes = line.as_bytes();
        let mut col = 0usize;
        while col + word.len() <= bytes.len() {
            if &bytes[col..col + word.len()] == word.as_bytes() {
                let before_ok = col == 0 || !is_word_byte(bytes[col - 1]);
                let after = col + word.len();
                let after_ok = after == bytes.len() || !is_word_byte(bytes[after]);
                if before_ok && after_ok {
                    out.push(json!({
                        "start": { "line": line_idx as u32, "character": col as u32 },
                        "end":   { "line": line_idx as u32, "character": (col + word.len()) as u32 },
                    }));
                    col += word.len();
                    continue;
                }
            }
            col += 1;
        }
    }
    out
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn is_valid_identifier(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.is_empty() { return false; }
    let first = bytes[0];
    if !(first.is_ascii_alphabetic() || first == b'_') { return false; }
    bytes[1..].iter().all(|&b| is_word_byte(b))
}

/// Convert a compiler `Span` (1-based line/col + byte offsets) into
/// a one-token-wide LSP range. Uses the token length passed in —
/// `span.end - span.start` would also work for simple identifiers.
fn span_to_range(_src: &str, span: &Span, len: u32) -> Value {
    let line = span.line.saturating_sub(1);
    let start = span.col.saturating_sub(1);
    json!({
        "start": { "line": line, "character": start },
        "end":   { "line": line, "character": start + len },
    })
}

/// Find the identifier-like word around a (line, col) position.
/// 0-based line and 0-based UTF-16 code unit column (LSP spec).
fn word_at_position(src: &str, line: u32, col: u32) -> String {
    let Some(line_text) = src.lines().nth(line as usize) else { return String::new(); };
    let bytes = line_text.as_bytes();
    let c = col as usize;
    if c > bytes.len() { return String::new(); }
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    // Walk left and right to find the word boundary.
    let mut start = c;
    while start > 0 && is_word(bytes[start - 1]) { start -= 1; }
    let mut end = c;
    while end < bytes.len() && is_word(bytes[end]) { end += 1; }
    if start == end { return String::new(); }
    line_text[start..end].to_string()
}

// --- Pipeline → LSP diagnostics -----------------------------------------

/// Run lex/parse/typecheck on `src`. Returns LSP diagnostics plus
/// (on each successful phase) the AST + ModuleInfo the hover handler
/// uses to resolve symbol types.
fn run_pipeline(
    src: &str,
    workspace_root: &Path,
) -> (Vec<Value>, Option<ast::Module>, Option<ModuleInfo>) {
    let tokens = match lexer::lex(src) {
        Ok(t) => t,
        Err(e) => return (vec![diagnostic_from_span(&e.span, &e.message, "lex")], None, None),
    };
    let module = match parser::parse(tokens) {
        Ok(m) => m,
        Err(e) => return (vec![diagnostic_from_span(&e.span, &e.message, "parse")], None, None),
    };
    let resolved = imports::resolve(&module, workspace_root);
    match types::typecheck(&module, &resolved.imported) {
        Ok(info) => (Vec::new(), Some(module), Some(info)),
        Err(errors) => {
            let diags = errors.iter()
                .map(|e: &TypeError| diagnostic_from_span(&e.span, &e.message, "type"))
                .collect();
            // Keep the parsed AST even if typecheck failed — hover
            // can still fall back to syntactic info (param types
            // from annotations, etc.).
            (diags, Some(module), None)
        }
    }
}

/// Look up hover info for `word`, preferring the most concrete source:
///   1. A top-level fn signature from the last good typecheck.
///   2. A user-defined struct / sum type.
///   3. A param or let/var in the AST with that name.
///
/// Returns a markdown-ready snippet.
fn resolve_hover(word: &str, doc: &DocState) -> Option<String> {
    if let Some(info) = &doc.last_info {
        if let Some(sig) = info.fns.get(word) {
            return Some(format_fn_signature(word, sig));
        }
        if let Some(ty) = info.types.get(word) {
            return Some(format_type_decl(word, ty));
        }
    }
    if let Some(module) = &doc.last_ast {
        if let Some(s) = find_binding_in_module(module, word) {
            return Some(s);
        }
    }
    None
}

fn format_fn_signature(name: &str, sig: &FnSig) -> String {
    let params: Vec<String> = sig.params.iter()
        .map(|(n, t)| format!("{n}: {}", t.display()))
        .collect();
    let generics = if sig.type_params.is_empty() {
        String::new()
    } else {
        format!("<{}>", sig.type_params.join(", "))
    };
    let effect = match sig.effect {
        crate::ast::Effect::Io => " io",
        crate::ast::Effect::Pure => "",
    };
    format!("```lumen\nfn {name}{generics}({}): {}{effect}\n```",
        params.join(", "),
        sig.ret.display())
}

fn format_type_decl(name: &str, info: &types::TypeInfo) -> String {
    use types::TypeInfo;
    match info {
        TypeInfo::Struct { fields, .. } => {
            let body: Vec<String> = fields.iter()
                .map(|(n, t)| format!("    {n}: {}", t.display()))
                .collect();
            format!("```lumen\ntype {name} = {{\n{}\n}}\n```", body.join(",\n"))
        }
        TypeInfo::Sum { variants, .. } => {
            let names: Vec<&str> = variants.iter().map(|v| v.name.as_str()).collect();
            format!("```lumen\ntype {name} = {}\n```", names.join(" | "))
        }
    }
}

/// Walk every function decl in the module; if any param or let/var
/// binding has the given name, format a one-line hover entry.
/// Later bindings shadow earlier ones — we return the first match
/// from a depth-first walk (good enough for MVP; position-aware
/// scoping is a follow-up).
fn find_binding_in_module(module: &ast::Module, word: &str) -> Option<String> {
    use ast::{Item, StmtKind};
    for item in &module.items {
        if let Item::Fn(f) = item {
            for p in &f.params {
                if p.name == word {
                    return Some(format!("```lumen\n(param) {}: {}\n```", p.name, display_ast_type(&p.ty)));
                }
            }
            // Walk statements for matching bindings.
            for stmt in &f.body.stmts {
                match &stmt.kind {
                    StmtKind::Let { name, ty, .. } | StmtKind::Var { name, ty, .. } => {
                        if name == word {
                            let ty_str = ty.as_ref().map(display_ast_type)
                                .unwrap_or_else(|| "_".to_string());
                            return Some(format!("```lumen\nlet {name}: {ty_str}\n```"));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    None
}

/// Stringify an ast::Type without requiring a ModuleInfo. Approximates
/// the compiler's internal Ty::display for cases where we only have
/// syntactic type info (e.g. hover on a let binding before typecheck
/// reaches a successful pass).
fn display_ast_type(t: &ast::Type) -> String {
    use ast::TypeKind;
    match &t.kind {
        TypeKind::Named { name, args } => {
            if args.is_empty() {
                name.clone()
            } else {
                let inner: Vec<String> = args.iter().map(display_ast_type).collect();
                format!("{name}<{}>", inner.join(", "))
            }
        }
        TypeKind::Tuple(elems) => {
            let inner: Vec<String> = elems.iter().map(display_ast_type).collect();
            format!("({})", inner.join(", "))
        }
        TypeKind::FnPtr { params, ret } => {
            let p: Vec<String> = params.iter().map(display_ast_type).collect();
            format!("fn({}): {}", p.join(", "), display_ast_type(ret))
        }
    }
}

fn diagnostic_from_span(span: &Span, message: &str, source: &str) -> Value {
    // LSP uses 0-based line + character (UTF-16). Our spans are
    // 1-based line and 1-based column — subtract one on each.
    let line = span.line.saturating_sub(1);
    let start_col = span.col.saturating_sub(1);
    // span.end is a byte offset, not a column. Approximate the end
    // by using the same line with a +1 width — good enough for
    // highlighting the first token of the error.
    let end_col = start_col + 1;
    json!({
        "range": {
            "start": { "line": line, "character": start_col },
            "end":   { "line": line, "character": end_col },
        },
        "severity": 1, // 1 = Error
        "source": format!("lumen-{source}"),
        "message": message,
    })
}

// --- initialize response -----------------------------------------------

fn initialize_result() -> Value {
    json!({
        "capabilities": {
            // 1 = Full text sync; client re-sends the whole document
            // on every change. Incremental sync is a follow-up.
            "textDocumentSync": {
                "openClose": true,
                "change": 1,
                "save": { "includeText": false },
            },
            "hoverProvider": true,
            "definitionProvider": true,
            "referencesProvider": true,
            "renameProvider": true,
        },
        "serverInfo": {
            "name": "lumen-lsp",
            "version": env!("CARGO_PKG_VERSION"),
        }
    })
}

fn respond(id: Option<Value>, result: Value) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("jsonrpc".to_string(), Value::String("2.0".to_string()));
    if let Some(id) = id {
        obj.insert("id".to_string(), id);
    }
    obj.insert("result".to_string(), result);
    Value::Object(obj)
}
