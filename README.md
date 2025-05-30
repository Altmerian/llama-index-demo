# llama-demo

A collection of examples demonstrating the use of llama_index.

## Structure

```
llama-demo/
├── pyproject.toml         # Project configuration
├── README.md              # Project documentation
├── LICENSE                # License file
├── .gitignore             # Git ignore file
├── data/                  # Data directory with example documents
├── src/                   # Source directory
│   └── llama_demo/        # Main package
│       ├── __init__.py    # Package initialization
│       ├── cli.py         # Command-line interface
│       └── examples/      # Examples subpackage
│           ├── __init__.py
│           └── simple_query.py
├── tests/                 # Test directory
│   ├── __init__.py
│   └── test_simple_query.py
└── docs/                  # Documentation
    └── index.md
```

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
llama-demo simple-query
```

Or customize the query:

```bash
llama-demo simple-query --query "your custom query" --data-dir "your/data/directory"
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
