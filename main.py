from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

if __name__ == "__main__":
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query("find the proverb")
    print(response)
