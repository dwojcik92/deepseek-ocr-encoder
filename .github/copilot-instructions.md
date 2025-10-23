# GitHub Copilot Instructions for deepseek-ocr-encoder

## Project Overview

This is a Python package that provides an optimized encoder for DeepSeek-OCR vision tasks. The package is published to PyPI and uses automated semantic versioning.

## Semantic Release Configuration

This repository uses **Python Semantic Release** for automated versioning and publishing:

- Branch: `main`
- Version location: `pyproject.toml:project.version`
- Build command: `python -m build`
- Auto-upload to PyPI: enabled

## Commit Message Convention

**CRITICAL**: All commits MUST follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for semantic release to work correctly.

### Commit Message Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Commit Types

- **feat**: A new feature (triggers MINOR version bump, e.g., 0.1.0 → 0.2.0)
- **fix**: A bug fix (triggers PATCH version bump, e.g., 0.1.0 → 0.1.1)
- **docs**: Documentation only changes (no version bump)
- **style**: Code style changes (formatting, missing semicolons, etc., no version bump)
- **refactor**: Code refactoring without changing functionality (no version bump)
- **perf**: Performance improvements (triggers PATCH version bump)
- **test**: Adding or updating tests (no version bump)
- **chore**: Changes to build process, dependencies, or auxiliary tools (no version bump)
- **ci**: Changes to CI configuration files and scripts (no version bump)

### Breaking Changes

Add `BREAKING CHANGE:` in the commit body or append `!` after the type/scope to trigger a MAJOR version bump (e.g., 0.1.0 → 1.0.0):

```
feat!: remove support for Python 3.9

BREAKING CHANGE: Python 3.10+ is now required
```

### Examples of Correct Commit Messages

✅ **Good:**
```
feat: add support for batch encoding
fix: resolve memory leak in CUDA graph capture
docs: update installation instructions for uv
refactor: simplify position embedding caching
perf: optimize image preprocessing pipeline
chore: update transformers dependency constraints
```

❌ **Bad (will break semantic release):**
```
Added batch encoding support
Fixed memory leak
Update docs
Refactoring code
Better performance
```

## Issue and Pull Request Naming

### Issue Titles

Issue titles should be clear and actionable. When creating issues via Copilot, use descriptive titles that follow this pattern:

- **Bug reports**: "Fix: [clear description of the bug]"
- **Feature requests**: "Feature: [clear description of the feature]"
- **Documentation**: "Docs: [what needs to be documented]"
- **Enhancement**: "Enhancement: [what to improve]"

Examples:
- ✅ "Fix: CUDA graph fails on multi-batch inputs"
- ✅ "Feature: Add support for WebP image format"
- ✅ "Docs: Add API reference for capture_cudagraph method"
- ❌ "bug" (too vague)
- ❌ "doesn't work" (unclear)

### Pull Request Titles

PR titles MUST follow the same Conventional Commits format as commit messages, as they may be used in the release notes:

- ✅ "feat: add WebP image format support"
- ✅ "fix: resolve CUDA OOM error with large images"
- ✅ "docs: improve troubleshooting section"
- ❌ "Update code" (doesn't follow convention)
- ❌ "Various fixes" (too vague)

## Code Style Guidelines

- **Formatter**: Black (line length: 100)
- **Linter**: Ruff (Python 3.10+ syntax)
- **Type checking**: MyPy (optional type hints)
- **Python version**: 3.10+ (3.12+ in development)

## Package Manager

This project uses **uv** as the recommended package manager. When suggesting installation commands, always use:

- `uv add deepseek-ocr-encoder` (for installing from PyPI)
- `uv pip install .` (for installing from source)
- `uv pip install -e ".[dev]"` (for development installation)

## Dependencies

### Critical Constraint

The package requires `transformers>=4.30.0,<4.48.0`. This is due to breaking changes in transformers 4.48.0+ where `LlamaFlashAttention2` was removed. Always respect this constraint when suggesting dependency updates.

### Core Dependencies

- torch >= 2.4.0
- torchvision >= 0.15.0
- transformers >= 4.30.0, < 4.48.0
- pillow >= 9.0.0
- pymupdf >= 1.23.0 (for PDF support)
- addict >= 2.4.0 (required by DeepSeek-OCR model)

## Testing

Run tests with:
```bash
pytest
```

Ensure all tests pass before committing changes.

## Release Process

**DO NOT manually update the version in `pyproject.toml`.** Semantic Release handles this automatically based on commit messages.

The release process is automated via semantic-release:
1. Commits to `main` branch trigger semantic-release
2. Semantic-release analyzes commit messages since last release
3. Version is automatically bumped in `pyproject.toml`
4. Package is built with `python -m build`
5. Package is uploaded to PyPI automatically

## Important Reminders

1. ✅ Always use Conventional Commits format for all commits
2. ✅ Use `uv` for package management in examples and documentation
3. ✅ Respect the transformers version constraint (< 4.48.0)
4. ✅ Run tests before committing
5. ❌ Never manually edit the version in `pyproject.toml`
6. ❌ Never use vague commit messages like "update", "fix", "changes"
