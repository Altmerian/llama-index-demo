"""
Tests for simple_query module
"""

import unittest
from unittest.mock import MagicMock, patch

from llama_demo.examples.simple_query import run_simple_query
from llama_demo.utils import get_default_data_dir


class TestSimpleQuery(unittest.TestCase):
    """Test cases for simple_query module"""

    def test_get_default_data_dir(self):
        """Test the default data directory path"""
        data_dir = get_default_data_dir()
        # Convert Path object to string for assertion
        self.assertIn("data", str(data_dir))
        # The path should now point to the root data directory
        self.assertTrue(str(data_dir).endswith("data"))

    @patch("llama_demo.examples.simple_query.SimpleDirectoryReader")
    @patch("llama_demo.examples.simple_query.VectorStoreIndex")
    def test_run_simple_query_with_explicit_dir(self, mock_vector_index, mock_reader):
        # Setup mocks
        mock_docs = MagicMock()
        mock_reader.return_value.load_data.return_value = mock_docs

        mock_index = MagicMock()
        mock_vector_index.from_documents.return_value = mock_index

        mock_engine = MagicMock()
        mock_index.as_query_engine.return_value = mock_engine

        mock_response = MagicMock()
        mock_engine.query.return_value = mock_response

        # Call the function with explicit data_dir
        result = run_simple_query("test query", "test_data")

        # Assertions
        mock_reader.assert_called_once_with("test_data")
        mock_vector_index.from_documents.assert_called_once_with(mock_docs)
        mock_index.as_query_engine.assert_called_once()
        mock_engine.query.assert_called_once_with("test query")
        self.assertEqual(result, mock_response)

    @patch("llama_demo.examples.simple_query.get_default_data_dir")
    @patch("llama_demo.examples.simple_query.SimpleDirectoryReader")
    @patch("llama_demo.examples.simple_query.VectorStoreIndex")
    def test_run_simple_query_with_default_dir(
        self, mock_vector_index, mock_reader, mock_get_default
    ):
        # Setup mocks
        mock_get_default.return_value = "default_data_dir"

        mock_docs = MagicMock()
        mock_reader.return_value.load_data.return_value = mock_docs

        mock_index = MagicMock()
        mock_vector_index.from_documents.return_value = mock_index

        mock_engine = MagicMock()
        mock_index.as_query_engine.return_value = mock_engine

        mock_response = MagicMock()
        mock_engine.query.return_value = mock_response

        # Call the function with default data_dir
        result = run_simple_query("test query")

        # Assertions
        mock_get_default.assert_called_once()
        mock_reader.assert_called_once_with("default_data_dir")
        mock_vector_index.from_documents.assert_called_once_with(mock_docs)
        mock_index.as_query_engine.assert_called_once()
        mock_engine.query.assert_called_once_with("test query")
        self.assertEqual(result, mock_response)

    def test_run_simple_query_empty_query(self):
        """Test that an empty or whitespace query raises ValueError."""
        with self.assertRaises(ValueError):
            run_simple_query(query_text="")
        with self.assertRaises(ValueError):
            run_simple_query(query_text="   ")


if __name__ == "__main__":
    unittest.main()
