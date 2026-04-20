//! Transitive import resolution.
//!
//! Given a parsed module, walks its `import` list (and the imports of
//! each loaded module in turn) against a base directory, returning
//! the set of parsed `ParsedImport` ASTs plus a map of module name →
//! on-disk path. Any load/lex/parse failure is collected as a
//! human-readable string so callers can surface them without halting
//! the rest of the resolution.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::ast;
use crate::lexer;
use crate::parser;
use crate::types::ParsedImport;

pub struct ResolvedImports {
    pub imported: Vec<ParsedImport>,
    pub paths: HashMap<String, String>,
    pub errors: Vec<String>,
}

/// Resolve `module.imports` transitively against `base_dir`. `base_dir`
/// is typically the workspace root (where `std/` and `vendor/` live).
/// Uses the same path-segment-to-file mapping the CLI driver uses:
/// `import std/http` → `<base>/std/http.lm`.
pub fn resolve(module: &ast::Module, base_dir: &Path) -> ResolvedImports {
    let mut imported: Vec<ParsedImport> = Vec::new();
    let mut paths: HashMap<String, String> = HashMap::new();
    let mut errors: Vec<String> = Vec::new();
    let mut loaded: HashSet<String> = HashSet::new();

    let mut queue: Vec<(Vec<String>, String)> = module.imports.iter()
        .map(|imp| {
            let file_name = imp.path.last().cloned().unwrap_or_default();
            let reg_name = imp.alias.clone().unwrap_or_else(|| file_name.clone());
            (imp.path.clone(), reg_name)
        })
        .collect();

    while let Some((path_segs, reg_name)) = queue.pop() {
        if loaded.contains(&reg_name) { continue; }
        loaded.insert(reg_name.clone());

        let file_name = path_segs.last().cloned().unwrap_or_default();
        let mut mod_path: PathBuf = base_dir.to_path_buf();
        for seg in &path_segs[..path_segs.len().saturating_sub(1)] {
            mod_path = mod_path.join(seg);
        }
        mod_path = mod_path.join(format!("{file_name}.lm"));
        if !mod_path.exists() { continue; }

        let mod_src = match std::fs::read_to_string(&mod_path) {
            Ok(s) => s,
            Err(e) => { errors.push(format!("read {}: {e}", mod_path.display())); continue; }
        };
        let mod_tokens = match lexer::lex(&mod_src) {
            Ok(t) => t,
            Err(e) => { errors.push(format!("lex {}: {}", mod_path.display(), e.message)); continue; }
        };
        let mod_ast = match parser::parse(mod_tokens) {
            Ok(m) => m,
            Err(e) => { errors.push(format!("parse {}: {}", mod_path.display(), e.message)); continue; }
        };

        for imp in &mod_ast.imports {
            let dep = imp.path.last().cloned().unwrap_or_default();
            let dep_name = imp.alias.clone().unwrap_or_else(|| dep.clone());
            if !loaded.contains(&dep_name) {
                queue.push((imp.path.clone(), dep_name));
            }
        }
        paths.insert(reg_name.clone(), mod_path.to_string_lossy().into_owned());
        imported.push(ParsedImport { name: reg_name, module: mod_ast });
    }

    ResolvedImports { imported, paths, errors }
}
