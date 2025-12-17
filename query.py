from llama_index.core import VectorStoreIndex, SimpleDirectoryReader


def answer_query(query: str):
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print(response)
    return response.response
