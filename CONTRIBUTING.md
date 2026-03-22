# Contributing

Contributions are welcome — whether that's code, bug reports, or documentation.

## Issues and pull requests

- **Bug reports or feature requests**: open an issue on GitHub.
- **Code changes**: fork the repo, make your changes, and open a pull request against `main`.

To set up a local dev environment:

```bash
git clone https://github.com/younesStrittmatter/slurptuna.git
cd slurptuna
uv sync --group dev
```

## Documentation

Documentation contributions are just as valuable as code.
The docs live in the `docs/` folder as plain Markdown files — no special tooling knowledge needed.
To preview changes locally:

```bash
uv run mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser.
