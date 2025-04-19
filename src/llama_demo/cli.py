"""
Command-line interface for running llama_index examples
"""

import argparse
import sys

from llama_demo.examples.simple_query import get_default_data_dir, run_simple_query


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Run llama_index examples")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Simple query command
    simple_parser = subparsers.add_parser(
        "simple-query", help="Run a simple query against documents"
    )
    simple_parser.add_argument(
        "--data-dir",
        default=None,
        help=f"Directory containing documents (default: {get_default_data_dir()})",
    )
    simple_parser.add_argument(
        "--query", default="find the proverb", help="Query to run"
    )

    # Parse arguments and run the appropriate command
    args = parser.parse_args()

    if args.command == "simple-query":
        response = run_simple_query(args.data_dir, args.query)
        print(response)
    elif not args.command:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
