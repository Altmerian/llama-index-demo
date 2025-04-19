# llama-demo

A collection of examples demonstrating the use of llama_index.

## Installation

### Using UV (Recommended)

Install UV if you don't have it already:

```bash
curl -sSf https://install.ultraviolet.rs | sh
# or on Windows
powershell -c "irm install.ultraviolet.rs | iex"
```

Install the package in development mode:

```bash
uv pip install -e .
```

### Using pip

Alternatively, you can use regular pip:

```bash
pip install -e .
```

## Usage

Run a simple query example:

```bash
python -m llama_demo.cli simple-query
```

Or customize the query:

```bash
python -m llama_demo.cli simple-query --query "your custom query" --data-dir "your/data/directory"
```

## Development

### Creating a virtual environment with UV

```bash
uv venv
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate     # On Windows
```

### Installing development dependencies

```bash
uv pip install -e ".[dev]"
```

### Running tests

```bash
pytest
```

## Adding New Examples

To add new examples:

1. Create a new module in `src/llama_demo/examples/`
2. Add the example to the CLI in `src/llama_demo/cli.py`
3. Create tests in `tests/`
4. Document the example in `docs/` 