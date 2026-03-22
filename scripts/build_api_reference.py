#!/usr/bin/env python3
"""Build docs/api.md dynamically from Python modules in src/slurptuna."""

from pathlib import Path


def discover_modules(package_root: Path) -> list[str]:
    """Return importable module names under the given package root."""
    modules: list[str] = []
    for py_file in sorted(package_root.glob("*.py")):
        if py_file.name == "__main__.py":
            continue
        if py_file.name == "__init__.py":
            modules.append("slurptuna")
            continue
        modules.append(f"slurptuna.{py_file.stem}")
    return modules


def generate_api_md(api_file: Path, modules: list[str]) -> None:
    """Generate docs/api.md with mkdocstrings directives for discovered modules."""
    lines = [
        "# API Reference",
        "",
        "This page is auto-generated from modules in `src/slurptuna`.",
        "",
        "## Package",
        "",
        "::: slurptuna",
        "",
    ]

    submodules = [m for m in modules if m != "slurptuna"]
    if submodules:
        lines.extend(["## Modules", ""])
        for module in submodules:
            lines.extend([f"### {module}", "", f"::: {module}", ""])

    api_file.write_text("\n".join(lines))
    print(f"Generated {api_file} from {len(modules)} module(s)")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    package_root = project_root / "src" / "slurptuna"
    api_file = project_root / "docs" / "api.md"

    if not package_root.exists():
        raise FileNotFoundError(f"Package path not found: {package_root}")

    modules = discover_modules(package_root)
    generate_api_md(api_file, modules)
