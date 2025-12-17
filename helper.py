# Add your utilities or helper functions to this file.

import os
from dotenv import load_dotenv, find_dotenv

# these expect to find a .env file at the directory above the lesson.
# the format for that file is (without the comment)
#API_KEYNAME=AStringThatIsTheLongAPIKeyFromSomeService 


from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PagedCSVReader
import re

from llama_index.llms.openai import OpenAI

# Initialize OpenAI LLM (ensure OPENAI_API_KEY is set in the environment)
llm = OpenAI(model='gpt-4', temperature=0.0)

def lookup_cost_from_csv(message: str, index, query_engine) -> float | None:
    """
    Extract make, model, year from message, query the indexed CSV, and return the cost.
    """
    # Simple parsing: two words + 4-digit year
    m = re.search(r'([A-Za-z]+)\s+([A-Za-z]+)\s+(\d{4})', message)
    if not m:
        return None
    make, model, year = m.group(1), m.group(2), m.group(3)
    # Query the index (RAG search)
    query = f"{make} {model} {year}"
    response = query_engine.query(query)
    text = str(response)  # Convert response to text
    # Parse the cost line, e.g. "Cost: 13000"
    cost_match = re.search(r'Cost[: ]+(\d+(\.\d+)?)', text)
    if cost_match:
        return float(cost_match.group(1))
    return None


def parse_offer(message: str) -> float | None:
    """
    Extract a numeric price from the message (e.g. "$25k", "15000").
    Returns the price as a float, or None if no price found.
    """
    pattern = re.compile(r'\$?\s*(\d[\d,]*(?:\.\d+)?)\s*([kK]?)')
    match = pattern.search(message)
    if not match:
        return None
    num_str, suffix = match.group(1), match.group(2)
    # Remove commas and convert to float
    amount = float(num_str.replace(",", ""))
    # Handle 'k' or 'K' suffix for thousands
    if suffix and suffix.lower() == 'k':
        amount *= 1000
    return amount


def answer_query(index, query):
    # Create a query engine that returns a response rooted in the indexed CSV data
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)
    response = query_engine.query(query)
    return response.response  # the LLM’s answer text


def load_csv_documents(csv_path):
    # Parse CSV: each row becomes one Document with "Column: Value" lines
    reader = PagedCSVReader()
    documents = reader.load_data(csv_path)
    return documents

def build_faiss_index(documents):
    # Create a FAISS vector store (using default FlatL2 index)
    faiss_store = FaissVectorStore(faiss_index=None)  # automatically creates an IndexFlatL2
    storage_context = StorageContext.from_defaults(vector_store=faiss_store)
    # Use OpenAI embeddings for vectors (requires OPENAI_API_KEY)
    embed_model = OpenAIEmbedding()
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
    return index

MIN_MARGIN = 0.10  # 10%

def negotiate_price(cost_price, offer_price):
    min_price = cost_price * (1 + MIN_MARGIN)
    if offer_price < min_price:
        # Offer too low - refuse
        return False, f"I'm sorry, we cannot accept ${offer_price:.2f}; our minimum acceptable price is ${min_price:.2f}."
    else:
        # Accept or counter
        return True, f"Your offer of ${offer_price:.2f} meets our margin requirements."

def check_need_human(query, retrieved_docs):
    # Simple heuristic: if query mentions 'price' (negotiation) or no docs found, flag it
    if "price" in query.lower() or not retrieved_docs:
        return True
    return False




def load_env():
    _ = load_dotenv(find_dotenv())

def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key

def get_llama_cloud_api_key():
    load_env()
    llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    return llama_cloud_api_key

def extract_html_content(filename):
    try:
        with open(filename, 'r') as file:
            html_content = file.read()
            html_content = f""" <div style="width: 100%; height: 800px; overflow: hidden;"> {html_content} </div>"""
            return html_content
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")