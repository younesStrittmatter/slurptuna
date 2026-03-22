#!/usr/bin/env python3
"""
Generate API reference documentation from docstrings using mkdocstrings directives.
Also ensures API Reference and Examples are in mkdocs navigation.
"""

from pathlib import Path
import yaml


def generate_api_md(api_file: Path) -> None:
    """Generate api.md with mkdocstrings directives."""
    content = """# API Reference

Core user-facing API for slurptuna.

## Defining Losses

:::python
slurptuna.loss

## Optimization Functions

:::python
slurptuna.optimize_run

:::python
slurptuna.optimize_entries

## Result Types

:::python
slurptuna.OptimizeResult

:::python
slurptuna.MultiOptimizeResult

## Parameter Specifications

:::python
slurptuna.search_param

:::python
slurptuna.SearchParam

## Execution Modes

:::python
slurptuna.ExecutionMode
"""
    api_file.write_text(content)
    print(f"Generated {api_file}")


def update_mkdocs_nav(mkdocs_file: Path) -> None:
    """Ensure api.md and examples.md are in mkdocs.yml nav."""
    if not mkdocs_file.exists():
        print(f"Error: {mkdocs_file} not found")
        return
    
    try:
        content = mkdocs_file.read_text()
        config = yaml.safe_load(content)
        
        if 'nav' not in config:
            config['nav'] = []
        
        nav = config['nav']
        
        # Define required entries in order
        nav_entries = [
            {'Examples': 'examples.md'},
            {'API Reference': 'api.md'},
        ]
        
        # Find where to insert (after Participant-wise Fitting)
        insert_idx = None
        for i, item in enumerate(nav):
            if isinstance(item, dict) and 'Participant-wise Fitting' in item:
                insert_idx = i + 1
                break
        
        # Add missing entries
        for entry in nav_entries:
            entry_title = list(entry.keys())[0]
            entry_file = entry[entry_title]
            
            # Check if entry already exists
            entry_exists = any(
                (isinstance(item, dict) and entry_file in str(item.values()))
                for item in nav
            )
            
            if not entry_exists:
                if insert_idx is not None:
                    nav.insert(insert_idx, entry)
                    insert_idx += 1
                else:
                    nav.append(entry)
                print(f"Added '{entry_title}' to mkdocs.yml nav")
        
        # Write back
        output = yaml.dump(config, default_flow_style=False, sort_keys=False)
        mkdocs_file.write_text(output)
        print(f"Updated {mkdocs_file}")
    except Exception as e:
        print(f"Error: Could not update mkdocs.yml: {e}")
        raise


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    api_file = project_root / 'docs' / 'api.md'
    mkdocs_file = project_root / 'mkdocs.yml'
    
    generate_api_md(api_file)
    update_mkdocs_nav(mkdocs_file)
