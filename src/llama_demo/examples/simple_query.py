"""
Simple query example using llama_index
"""

from pathlib import Path
from typing import Union

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.response.schema import Response

from llama_demo.utils import get_default_data_dir


def run_simple_query(
    query_text: str, data_dir: Union[Path, str, None] = None
) -> Response:
    """
    Run a simple query against documents in the specified directory

    Args:
        query_text: The query to run against the index
        data_dir: Directory containing documents to index (str or Path)

    Returns:
        Query response

    Raises:
        ValueError: If query_text is empty or contains only whitespace.
    """
    if not query_text or query_text.isspace():
        raise ValueError("Query text cannot be empty or whitespace.")

    if data_dir is None:
        data_dir = get_default_data_dir()

    # SimpleDirectoryReader accepts Path objects, but let's ensure it's a string
    # if compatibility is uncertain or for consistency.
    documents = SimpleDirectoryReader(
        str(data_dir), exclude=["paul_graham_essay.txt"]
    ).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    return query_engine.query(query_text)


def main() -> None:
    """Run the simple query example as a standalone script"""
    # Example usage: Call run_simple_query with a specific query
    # response = run_simple_query(query_text="Your query here")
    # print(response)
    print("Simple query module executed. Use CLI or call run_simple_query directly.")


if __name__ == "__main__":
    main()
