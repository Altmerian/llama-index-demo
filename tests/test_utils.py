"""
Tests for utility functions.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

from llama_demo.utils import find_project_root, get_default_data_dir


class TestUtils(unittest.TestCase):
    """Test cases for utility functions"""

    def test_find_project_root(self):
        """Test finding the project root."""
        project_root = find_project_root()
        self.assertIsInstance(project_root, Path)
        # Check if the marker file exists in the found root
        self.assertTrue((project_root / "pyproject.toml").exists())

    def test_find_project_root_not_found(self):
        """Test behavior when the project root marker is not found."""
        # Mock Path.exists to always return False
        with patch("pathlib.Path.exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                find_project_root()

    def test_get_default_data_dir(self):
        """Test getting the default data directory path."""
        project_root = find_project_root()  # Assuming this works from the other test
        data_dir = get_default_data_dir()
        self.assertIsInstance(data_dir, Path)
        self.assertEqual(data_dir.name, "data")
        self.assertEqual(data_dir.parent, project_root)
        # Check relative path construction
        self.assertEqual(data_dir, project_root / "data")


if __name__ == "__main__":
    unittest.main()
