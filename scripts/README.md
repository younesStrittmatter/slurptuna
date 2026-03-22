# Documentation Build System

This directory contains scripts for automatically generating slurptuna documentation.

## Overview

The documentation build system includes two automation scripts:

1. **`build_examples_docs.py`** - Parses example files and auto-generates `docs/examples.md`
2. **`build_api_reference.py`** - Discovers modules in `src/slurptuna` and auto-generates `docs/api.md`

Plus the Makefile orchestrates the full build process.

## How It Works

### Example Documentation Generation

The `build_examples_docs.py` script:

1. **Scans** `examples/` directory for `run_*.py` files
2. **Parses** each file using AST to extract:
   - `@loss` decorator metadata (name, description)
   - Single-mode code examples (commented sections)
   - Distributed-mode code examples (active code)
3. **Categorizes** examples (basic, average, individual)
4. **Generates** `docs/examples.md` with:
   - Organized sections by use case
   - Both local and distributed code examples
   - Links back to source files

### API Reference Generation

The `build_api_reference.py` script:

1. Scans `src/slurptuna/*.py` to discover package modules dynamically
2. Generates `docs/api.md` with mkdocstrings directives for each discovered module
3. Keeps API docs in sync with the codebase without hardcoded symbol lists

### Building Documentation

```bash
# Generate examples + API reference + build mkdocs site
make docs-build

# Generate examples + API reference + serve live preview
make docs-serve

# Just generate examples.md (without full build)
python scripts/build_examples_docs.py

# Just regenerate API reference (without full build)
python scripts/build_api_reference.py
```

## Adding New Examples

To add a new example to the documentation:

1. Create `run_your_example.py` in the `examples/` directory
2. Include `@loss` decorator with `name` and `description`
3. Include both `# LOCAL` and `# DISTRIBUTED` example code in `__main__`
4. Run `make docs-build` or just `python scripts/build_examples_docs.py`

The documentation will automatically update with your new example!

### Example Template

```python
from slurptuna import loss, optimize_run

@loss(
    name="your_model",
    description="A short description of what this example demonstrates",
    parameter_space={"param1": (0.0, 1.0)},
)
def your_model(params, seed):
    return loss_value

if __name__ == "__main__":
    # LOCAL / single-mode example.
    # result = optimize_run(
    #     your_model,
    #     mode=ExecutionMode.SINGLE,
    #     n_trials=20,
    #     n_seeds=200,
    # )

    # DISTRIBUTED example.
    result = optimize_run(
        your_model,
        mode=ExecutionMode.DISTRIBUTED,
        n_trials=6,
        n_seeds=120,
        chunk_size=20,
    )
    print(result.best_params)
```

## Implementation Notes

- The scripts use Python's `ast` module to safely parse files without executing them
- Regex patterns extract code blocks from `if __name__ == "__main__":` sections  
- Indentation is handled automatically during code block extraction
- Examples are sorted alphabetically and categorized for better organization
- The generated `examples.md` includes usage instructions and result interpretation
- API module discovery is automatic and based on files under `src/slurptuna`

## Files

- `build_examples_docs.py` - Generates examples.md from example files
- `build_api_reference.py` - Generates api.md from discovered source modules
- `../Makefile` - Build orchestration (in project root)
