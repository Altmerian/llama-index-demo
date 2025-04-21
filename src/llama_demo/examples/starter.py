import asyncio

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

from llama_demo.utils import get_default_data_dir

# Create a RAG tool using LlamaIndex
documents = SimpleDirectoryReader(
    input_files=[get_default_data_dir() / "paul_graham_essay.txt"]
).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()


# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about an personal essay written by Paul Graham."""
    response = await query_engine.aquery(query)
    return str(response)


# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    tools=[multiply, search_documents],
    llm=OpenAI(model="gpt-4.1-nano-2025-04-14"),
    system_prompt="""
    You are a helpful assistant that can multiply two numbers 
    and search through documents to answer questions.""",
)


async def main():
    ctx = Context(workflow=agent)

    # run agent with context
    await agent.run("My name is Logan", ctx=ctx)
    response = await agent.run("What is my name?", ctx=ctx)
    print(str(response))

    # Run the agent
    response = await agent.run("What did the author do in college? Also, what's 7 * 8?")
    print(str(response))


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
