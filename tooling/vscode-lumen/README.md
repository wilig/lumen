# Lumen for VS Code

Syntax highlighting and basic editor support for the [Lumen](https://github.com/wilig/lumen) programming language.

## Features

- Syntax highlighting for `.lm` files
- Bracket matching and auto-closing
- Comment toggling (`Ctrl+/`)
- String interpolation `\{expr}` highlighted as embedded code

## Install (development)

From this directory, symlink the extension into your VS Code extensions folder:

```sh
# Linux / macOS
ln -s "$PWD" ~/.vscode/extensions/lumen-0.1.0

# Then reload VS Code (Ctrl+Shift+P → "Developer: Reload Window")
```

Or package and install via `vsce`:

```sh
npm install -g @vscode/vsce
vsce package
code --install-extension lumen-0.1.0.vsix
```

## Files

- `package.json` — extension manifest
- `language-configuration.json` — bracket pairs, comment toggle, indent rules
- `syntaxes/lumen.tmLanguage.json` — TextMate grammar

## Tree-sitter

A tree-sitter grammar for Lumen (targeting Neovim, Helix, Zed, Emacs ts-mode) is tracked separately and not yet implemented.
