#!/usr/bin/env python3
"""
Automatically generate examples.md documentation by parsing example files
from the examples/ directory.
"""

import ast
import re
from pathlib import Path


def extract_loss_info(file_path: Path) -> dict:
    """Extract @loss decorator info and file content from an example file."""
    content = file_path.read_text()
    
    # Parse AST to find @loss decorator
    tree = ast.parse(content)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check if it has @loss decorator
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    if (isinstance(decorator.func, ast.Name) and 
                        decorator.func.id == 'loss'):
                        
                        # Extract decorator kwargs
                        loss_name = None
                        description = None
                        
                        for keyword in decorator.keywords:
                            if keyword.arg == 'name':
                                if isinstance(keyword.value, ast.Constant):
                                    loss_name = keyword.value.value
                            elif keyword.arg == 'description':
                                if isinstance(keyword.value, ast.Constant):
                                    description = keyword.value.value
                        
                        # Extract function docstring if present
                        docstring = ast.get_docstring(node)
                        
                        return {
                            'name': loss_name or node.name,
                            'description': description or docstring or '',
                            'function_name': node.name,
                            'content': content,
                        }
    
    return None


def extract_code_blocks(content: str) -> tuple[str, str]:
    """Extract single-mode and distributed-mode code blocks from __main__."""
    # Find the if __name__ == "__main__": block  
    match = re.search(
        r'if __name__ == "__main__":\n(.*)',
        content,
        re.DOTALL
    )
    if not match:
        return "", ""
    
    full_code = match.group(1)
    
    # Find indices of markers
    local_pattern = r'# LOCAL.*?result\s*=\s*optimize'
    dist_pattern = r'# DISTRIBUTED.*?result\s*=\s*optimize'
    
    local_match = re.search(local_pattern, full_code, re.DOTALL)
    dist_match = re.search(dist_pattern, full_code, re.DOTALL)
    
    single_block = ""
    dist_block = ""
    
    # Extract single-mode block
    if local_match:
        start = local_match.start()
        # Find where this block ends (before DISTRIBUTED or end of file)
        dist_start = full_code.find('# DISTRIBUTED')
        if dist_start == -1:
            end = len(full_code)
        else:
            end = dist_start
        
        single_text = full_code[start:end]
        # Dedent
        lines = single_text.split('\n')
        min_indent = float('inf')
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)
        if min_indent == float('inf'):
            min_indent = 0
        single_block = '\n'.join(
            line[min_indent:] if line.strip() else line
            for line in lines
        ).strip()
    
    # Extract distributed-mode block
    if dist_match:
        start = dist_match.start()
        end = len(full_code)
        
        dist_text = full_code[start:end]
        # Dedent
        lines = dist_text.split('\n')
        min_indent = float('inf')
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)
        if min_indent == float('inf'):
            min_indent = 0
        dist_block = '\n'.join(
            line[min_indent:] if line.strip() else line
            for line in lines
        ).strip()
    
    return single_block, dist_block


def categorize_example(filename: str) -> tuple[str, str]:
    """Categorize example and return (category, title)."""
    if 'toy' in filename:
        return ('basic', 'Basic Optimization (Toy Loss)')
    elif 'average' in filename:
        return ('average', 'Shared Fitting Across Participants')
    elif 'individual' in filename:
        return ('individual', 'Participant-wise Fitting')
    else:
        return ('other', filename.replace('run_', '').replace('.py', '').title())


def generate_examples_md(examples_dir: Path, output_file: Path) -> None:
    """Generate examples.md from example files."""
    
    # Discover example files
    example_files = sorted(examples_dir.glob('run_*.py'))
    
    if not example_files:
        print(f"No example files found in {examples_dir}")
        return
    
    # Parse examples
    examples = []
    for filepath in example_files:
        info = extract_loss_info(filepath)
        if info:
            category, title = categorize_example(filepath.name)
            single_code, dist_code = extract_code_blocks(info['content'])
            examples.append({
                'filename': filepath.name,
                'category': category,
                'title': title,
                'loss_name': info['name'],
                'description': info['description'],
                'function_name': info['function_name'],
                'single_code': single_code,
                'dist_code': dist_code,
                'content': info['content'],
            })
    
    # Generate markdown
    md_sections = [
        "# Examples",
        "",
        "This page showcases common usage patterns with slurptuna.",
        "",
    ]
    
    for example in examples:
        md_sections.extend([
            f"## {example['title']}",
            "",
            example['description'],
            "",
        ])
        
        # Add use case sections for certain examples
        if example['category'] == 'average':
            md_sections.extend([
                "### Use Case",
                "When you want one set of parameters that works well for all participants, measuring success as the average loss across participants.",
                "",
            ])
        elif example['category'] == 'individual':
            md_sections.extend([
                "### Use Case",
                "When participants have different characteristics and you want individualized parameters for each (e.g., personalized model fits in psychology or medicine).",
                "",
            ])
        
        # Add single mode if available
        if example['single_code']:
            md_sections.extend([
                "### Single Mode (Local)",
                "",
                "```python",
            ])
            md_sections.extend(example['single_code'].split('\n'))
            md_sections.extend([
                "```",
                "",
            ])
        
        # Add distributed mode if available
        if example['dist_code']:
            md_sections.extend([
                "### Distributed Mode (Slurm Arrays)",
                "",
                "```python",
            ])
            md_sections.extend(example['dist_code'].split('\n'))
            md_sections.extend([
                "```",
                "",
            ])
        
        # Add file reference
        md_sections.extend([
            f"See [{example['filename']}](../examples/{example['filename']}) in the repository.",
            "",
        ])
    
    # Add results section
    md_sections.extend([
        "## Reading Results",
        "",
        "All runs produce:",
        "",
        "- **`summary.json`**: Best parameters and best value (updated progressively as trials complete)",
        "- **`meta.json`**: Run configuration and metadata",
        "- **`optuna.db`**: SQLite database with full trial history",
        "- **`trials/`**: Individual trial directories with chunk results",
        "",
        "Example:",
        "",
        "```python",
        "import json",
        "from pathlib import Path",
        "",
        "run_dir = Path(result.run_dir)",
        "summary = json.loads((run_dir / 'summary.json').read_text())",
        "print(f'Best value: {summary[\"best_value\"]}')",
        "print(f'Best params: {summary[\"best_params\"]}')",
        "```",
        "",
        "## More Information",
        "",
        "For details on distributed execution with Slurm, see [Distributed Mode](distributed.md).",
    ])
    
    # Write output
    output_content = '\n'.join(md_sections)
    output_file.write_text(output_content)
    print(f"Generated {output_file}")


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    examples_dir = project_root / 'examples'
    output_file = project_root / 'docs' / 'examples.md'
    
    generate_examples_md(examples_dir, output_file)
