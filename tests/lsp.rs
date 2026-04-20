// End-to-end LSP test: spawn `lumen lsp`, drive an initialize →
// didOpen → expect diagnostics sequence, verify responses.

use std::io::{BufRead, BufReader, Read as _, Write};
use std::process::{Command, Stdio};

/// Frame a JSON body with the LSP Content-Length header.
fn frame(body: &str) -> String {
    format!("Content-Length: {}\r\n\r\n{body}", body.len())
}

/// Read one LSP message (headers + body) from a reader and return
/// the raw JSON body string.
fn read_message(reader: &mut impl BufRead) -> String {
    let mut content_length = 0usize;
    let mut header = String::new();
    loop {
        header.clear();
        reader.read_line(&mut header).unwrap();
        let line = header.trim_end_matches(['\r', '\n']);
        if line.is_empty() { break; }
        if let Some(rest) = line.strip_prefix("Content-Length:") {
            content_length = rest.trim().parse().unwrap();
        }
    }
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body).unwrap();
    String::from_utf8(body).unwrap()
}

#[test]
fn lsp_initialize_then_diagnostics_on_type_error() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_lumen"))
        .arg("lsp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn lumen lsp");

    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);

    // 1. initialize
    let init = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}"#;
    stdin.write_all(frame(init).as_bytes()).unwrap();
    let resp = read_message(&mut reader);
    assert!(resp.contains("textDocumentSync"),
        "initialize response missing capabilities:\n{resp}");
    assert!(resp.contains("hoverProvider"),
        "initialize response missing hoverProvider:\n{resp}");

    // 2. initialized notification (no response)
    let initialized = r#"{"jsonrpc":"2.0","method":"initialized","params":{}}"#;
    stdin.write_all(frame(initialized).as_bytes()).unwrap();

    // 3. Open a document with a type error: return 1 where string is expected.
    let src = r#"fn greet(): string {\n    return 1\n}"#;
    let did_open = format!(
        r#"{{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{{"textDocument":{{"uri":"file:///t.lm","languageId":"lumen","version":1,"text":"{}"}}}}}}"#,
        src,
    );
    stdin.write_all(frame(&did_open).as_bytes()).unwrap();

    // 4. Expect a publishDiagnostics notification with a type error.
    let diag = read_message(&mut reader);
    assert!(diag.contains("publishDiagnostics"),
        "expected publishDiagnostics, got:\n{diag}");
    assert!(diag.contains("lumen-type"),
        "expected a type diagnostic, got:\n{diag}");

    // 5. shutdown / exit
    let shutdown = r#"{"jsonrpc":"2.0","id":2,"method":"shutdown"}"#;
    stdin.write_all(frame(shutdown).as_bytes()).unwrap();
    let _ = read_message(&mut reader);
    let exit = r#"{"jsonrpc":"2.0","method":"exit"}"#;
    stdin.write_all(frame(exit).as_bytes()).unwrap();
    drop(stdin);

    let status = child.wait().expect("wait on lsp");
    assert!(status.success(), "lsp exited with {status}");
}

#[test]
fn lsp_resolves_std_imports() {
    // With workspace-wide import resolution, a document that imports
    // std/io and calls io.println should typecheck cleanly. Without
    // it, io would be unknown and the call would fail.
    let mut child = Command::new(env!("CARGO_BIN_EXE_lumen"))
        .arg("lsp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn lumen lsp");

    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);

    // Initialize WITHOUT a rootUri — the server should fall back to
    // the compiler source tree (CARGO_MANIFEST_DIR) so std/ is still
    // visible.
    let init = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}"#;
    stdin.write_all(frame(init).as_bytes()).unwrap();
    let _ = read_message(&mut reader);

    // `\n` must be a JSON-escaped newline inside the text string.
    let src = r#"import std/io\n\nfn main(): i32 io {\n    io.println(\"hi\")\n    return 0\n}"#;
    let did_open = format!(
        r#"{{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{{"textDocument":{{"uri":"file:///imp.lm","languageId":"lumen","version":1,"text":"{}"}}}}}}"#,
        src,
    );
    stdin.write_all(frame(&did_open).as_bytes()).unwrap();

    let diag = read_message(&mut reader);
    assert!(diag.contains(r#""diagnostics":[]"#),
        "std/io-using document should typecheck clean, got:\n{diag}");

    let shutdown = r#"{"jsonrpc":"2.0","id":2,"method":"shutdown"}"#;
    stdin.write_all(frame(shutdown).as_bytes()).unwrap();
    let _ = read_message(&mut reader);
    let exit = r#"{"jsonrpc":"2.0","method":"exit"}"#;
    stdin.write_all(frame(exit).as_bytes()).unwrap();
    drop(stdin);
    let _ = child.wait();
}

#[test]
fn lsp_hover_shows_fn_signature() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_lumen"))
        .arg("lsp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn lumen lsp");
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);

    let init = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}"#;
    stdin.write_all(frame(init).as_bytes()).unwrap();
    let _ = read_message(&mut reader);

    // Define `greet(name: string): string`, then hover on `greet`
    // inside its own body.
    let src = r#"fn greet(name: string): string {\n    return name\n}"#;
    let did_open = format!(
        r#"{{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{{"textDocument":{{"uri":"file:///h.lm","languageId":"lumen","version":1,"text":"{}"}}}}}}"#,
        src,
    );
    stdin.write_all(frame(&did_open).as_bytes()).unwrap();
    let _ = read_message(&mut reader); // diagnostics

    // Position cursor inside `greet` (line 0, char 4).
    let hover = r#"{"jsonrpc":"2.0","id":10,"method":"textDocument/hover","params":{"textDocument":{"uri":"file:///h.lm"},"position":{"line":0,"character":4}}}"#;
    stdin.write_all(frame(hover).as_bytes()).unwrap();
    let resp = read_message(&mut reader);
    assert!(resp.contains("fn greet(name: string)"),
        "hover should show greet's signature, got:\n{resp}");

    // Hover on the `name` param at (line 0, char 9).
    let hover2 = r#"{"jsonrpc":"2.0","id":11,"method":"textDocument/hover","params":{"textDocument":{"uri":"file:///h.lm"},"position":{"line":0,"character":9}}}"#;
    stdin.write_all(frame(hover2).as_bytes()).unwrap();
    let resp2 = read_message(&mut reader);
    assert!(resp2.contains("(param) name: string"),
        "hover on param should show its declared type, got:\n{resp2}");

    let shutdown = r#"{"jsonrpc":"2.0","id":2,"method":"shutdown"}"#;
    stdin.write_all(frame(shutdown).as_bytes()).unwrap();
    let _ = read_message(&mut reader);
    stdin.write_all(frame(r#"{"jsonrpc":"2.0","method":"exit"}"#).as_bytes()).unwrap();
    drop(stdin);
    let _ = child.wait();
}

#[test]
fn lsp_clean_document_has_empty_diagnostics() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_lumen"))
        .arg("lsp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn lumen lsp");

    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);

    let init = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}"#;
    stdin.write_all(frame(init).as_bytes()).unwrap();
    let _ = read_message(&mut reader);

    let src = r#"fn main(): i32 {\n    return 0\n}"#;
    let did_open = format!(
        r#"{{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{{"textDocument":{{"uri":"file:///ok.lm","languageId":"lumen","version":1,"text":"{}"}}}}}}"#,
        src,
    );
    stdin.write_all(frame(&did_open).as_bytes()).unwrap();

    let diag = read_message(&mut reader);
    assert!(diag.contains(r#""diagnostics":[]"#),
        "clean document should have no diagnostics, got:\n{diag}");

    let shutdown = r#"{"jsonrpc":"2.0","id":2,"method":"shutdown"}"#;
    stdin.write_all(frame(shutdown).as_bytes()).unwrap();
    let _ = read_message(&mut reader);
    let exit = r#"{"jsonrpc":"2.0","method":"exit"}"#;
    stdin.write_all(frame(exit).as_bytes()).unwrap();
    drop(stdin);
    let _ = child.wait();
}
