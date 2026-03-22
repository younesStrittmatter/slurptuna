#!/usr/bin/env python3
"""Build docs/api.md dynamically from Python modules in src/slurptuna."""

import ast
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


def module_file_for(module_name: str, package_root: Path) -> Path:
    if module_name == "slurptuna":
        return package_root / "__init__.py"
    return package_root / f"{module_name.rsplit('.', 1)[-1]}.py"


def public_members(module_file: Path) -> list[str]:
    """Return public top-level classes/functions, or __all__ if defined."""
    tree = ast.parse(module_file.read_text(), filename=str(module_file))

    exported: list[str] | None = None
    members: list[str] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                members.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        values: list[str] = []
                        for item in node.value.elts:
                            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                                values.append(item.value)
                        exported = values

    return exported if exported is not None else members


def directive_lines(module_name: str, members: list[str]) -> list[str]:
    lines = [f"::: {module_name}"]
    if members:
        lines.extend(
            [
                "    options:",
                "      show_root_heading: true",
                "      show_source: true",
                "      show_category_heading: true",
                "      members:",
            ]
        )
        for member in members:
            lines.append(f"        - {member}")
    else:
        lines.extend(
            [
                "    options:",
                "      show_root_heading: true",
                "      show_source: true",
                "      show_category_heading: true",
                "      members: []",
            ]
        )
    return lines


def generate_api_md(api_file: Path, package_root: Path, modules: list[str]) -> None:
    """Generate docs/api.md with mkdocstrings directives for discovered modules."""
    root_members = public_members(module_file_for("slurptuna", package_root))

    lines = [
        "# API Reference",
        "",
        "This page is auto-generated from the public API exported by `slurptuna`.",
        "",
        "## Public API",
        "",
    ]
    lines.extend(directive_lines("slurptuna", root_members))
    lines.append("")

    api_file.write_text("\n".join(lines))
    print(f"Generated {api_file} from {len(root_members)} public export(s)")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    package_root = project_root / "src" / "slurptuna"
    api_file = project_root / "docs" / "api.md"

    if not package_root.exists():
        raise FileNotFoundError(f"Package path not found: {package_root}")

    modules = discover_modules(package_root)
    generate_api_md(api_file, package_root, modules)
