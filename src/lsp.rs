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

use serde_json::{json, Value};

use crate::lexer;
use crate::parser;
use crate::span::Span;
use crate::types::{self, ParsedImport, TypeError};

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
    /// Most recent source text for each open document, keyed by URI.
    docs: HashMap<String, String>,
    shutting_down: bool,
    exit_requested: bool,
}

impl Server {
    fn handle(&mut self, msg: Value) -> Vec<Value> {
        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let id = msg.get("id").cloned();
        let params = msg.get("params").cloned().unwrap_or(Value::Null);

        match method {
            "initialize" => vec![respond(id, initialize_result())],
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
        self.docs.insert(uri.to_string(), text);
        vec![self.diagnostics_for(uri)]
    }

    fn on_did_change(&mut self, params: &Value) -> Vec<Value> {
        let Some(uri) = text_document_uri(params) else { return Vec::new(); };
        // We advertise full-text sync, so each change carries the
        // complete document. Partial-sync support is a follow-up.
        let changes = params.pointer("/contentChanges").and_then(|c| c.as_array());
        if let Some(changes) = changes {
            if let Some(last) = changes.last() {
                if let Some(text) = last.get("text").and_then(|t| t.as_str()) {
                    self.docs.insert(uri.to_string(), text.to_string());
                }
            }
        }
        vec![self.diagnostics_for(uri)]
    }

    fn on_did_save(&mut self, params: &Value) -> Vec<Value> {
        let Some(uri) = text_document_uri(params) else { return Vec::new(); };
        vec![self.diagnostics_for(uri)]
    }

    fn on_hover(&self, params: &Value) -> Value {
        // MVP: report the symbol at the cursor position by its parsed
        // identifier. Full type-aware hover hooks into the typechecker
        // and is a clean follow-up.
        let Some(uri) = text_document_uri(params) else { return Value::Null; };
        let Some(src) = self.docs.get(uri) else { return Value::Null; };
        let line = params.pointer("/position/line").and_then(|l| l.as_u64()).unwrap_or(0) as u32;
        let col = params.pointer("/position/character").and_then(|c| c.as_u64()).unwrap_or(0) as u32;
        let word = word_at_position(src, line, col);
        if word.is_empty() {
            return Value::Null;
        }
        json!({
            "contents": {
                "kind": "markdown",
                "value": format!("`{word}`"),
            }
        })
    }

    /// Re-run lex / parse / typecheck for the document and convert
    /// any errors into LSP diagnostics. Emits a
    /// textDocument/publishDiagnostics notification (possibly with
    /// an empty list to clear prior errors).
    fn diagnostics_for(&self, uri: &str) -> Value {
        let src = self.docs.get(uri).cloned().unwrap_or_default();
        let diags = run_pipeline(&src);
        json!({
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": uri,
                "diagnostics": diags,
            }
        })
    }
}

fn text_document_uri(params: &Value) -> Option<&str> {
    params.pointer("/textDocument/uri").and_then(|u| u.as_str())
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

/// Run lex/parse/typecheck on `src` and return LSP `Diagnostic` values.
fn run_pipeline(src: &str) -> Vec<Value> {
    let tokens = match lexer::lex(src) {
        Ok(t) => t,
        Err(e) => return vec![diagnostic_from_span(&e.span, &e.message, "lex")],
    };
    let module = match parser::parse(tokens) {
        Ok(m) => m,
        Err(e) => return vec![diagnostic_from_span(&e.span, &e.message, "parse")],
    };
    // Typecheck with no imports resolved — LSP mode is single-file
    // for now. Full workspace resolution (walking imported modules
    // from the filesystem) is a follow-up.
    let imports: Vec<ParsedImport> = Vec::new();
    match types::typecheck(&module, &imports) {
        Ok(_) => Vec::new(),
        Err(errors) => errors.iter()
            .map(|e: &TypeError| diagnostic_from_span(&e.span, &e.message, "type"))
            .collect(),
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
