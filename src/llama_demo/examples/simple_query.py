"""
Simple query example using llama_index
"""

from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.response.schema import Response


def find_project_root(marker="pyproject.toml") -> Path:
    """Find the project root directory by searching upwards for a marker file."""
    current_dir = Path(__file__).resolve().parent
    while True:
        if (current_dir / marker).exists():
            return current_dir
        parent_dir = current_dir.parent
        if parent_dir == current_dir:
            # Reached filesystem root, maybe fallback or raise?
            # For now, let's raise a clearer error.
            raise FileNotFoundError(
                f"Project root marker '{marker}' not found starting from {Path(__file__).resolve().parent}"
            )
        current_dir = parent_dir


def get_default_data_dir() -> Path:
    """Get the default data directory path relative to the project root."""
    project_root = find_project_root()
    return project_root / "data"


def run_simple_query(data_dir=None, query_text="find the proverb") -> Response:
    """
    Run a simple query against documents in the specified directory

    Args:
        data_dir: Directory containing documents to index (str or Path)
        query_text: The query to run against the index

    Returns:
        Query response
    """
    if data_dir is None:
        data_dir = get_default_data_dir()

    # SimpleDirectoryReader accepts Path objects, but let's ensure it's a string
    # if compatibility is uncertain or for consistency.
    documents = SimpleDirectoryReader(str(data_dir)).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    return query_engine.query(query_text)


def main() -> None:
    """Run the simple query example as a standalone script"""
    response = run_simple_query()
    print(response)


if __name__ == "__main__":
    main()
