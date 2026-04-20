# tree-sitter-lumen

Tree-sitter grammar for the [Lumen](https://github.com/wgroppe/lumen) programming language.

Mirrors `docs/grammar.ebnf`. Targets modern editors (Neovim, Helix, Zed, Emacs
treesit) where tree-sitter is the default highlighter — more accurate than a
regex grammar and the structural tree enables folding, navigation, and
code-action targets.

## Layout

- `grammar.js` — grammar definition
- `queries/highlights.scm` — syntax highlight queries (standard capture names)
- `queries/injections.scm` — language injection points (placeholder)
- `test/corpus/` — parser regression tests
- `src/` — generated parser (written by `tree-sitter generate`; gitignored)

## Development

```bash
tree-sitter generate        # regenerate the parser from grammar.js
tree-sitter test            # run the corpus tests
tree-sitter parse FILE.lm   # parse a Lumen file and print the CST
```

Quick sanity check across the repo's real Lumen code:

```bash
for f in examples/*.lm std/*.lm tests/**/*.lm; do
  tree-sitter parse -q "$f" > /dev/null || echo "fail: $f"
done
```

## Editor integration

Neovim (nvim-treesitter) — add a parser config in `init.lua`:

```lua
require('nvim-treesitter.parsers').get_parser_configs().lumen = {
  install_info = {
    url = 'path/to/tooling/tree-sitter-lumen',
    files = {'src/parser.c'},
    branch = 'main',
  },
  filetype = 'lumen',
}
```

Helix — add to `languages.toml`:

```toml
[[language]]
name = "lumen"
scope = "source.lumen"
file-types = ["lm"]
roots = []
comment-token = "//"

[[grammar]]
name = "lumen"
source = { path = "path/to/tooling/tree-sitter-lumen" }
```

## When the grammar diverges

If `src/parser.rs` (the Rust parser) disagrees with `docs/grammar.ebnf`,
the Rust parser is authoritative. Fix the EBNF and this grammar in
lockstep when that happens.
