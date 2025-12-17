import asyncio
from pathlib import Path
from typing import List, Tuple
from pydantic import BaseModel, Field
from llama_index.core import VectorStoreIndex
from llama_index.readers.file import CSVReader
from llama_index.llms.openai import OpenAI


class SalesState(BaseModel):
    """Tracks ongoing session information."""
    query_count: int = Field(default=0)
    last_question: str = Field(default="")
    last_answer: str = Field(default="")
    chat_history: List[Tuple[str, str]] = Field(default_factory=list)  # [(Q, A), (Q, A), ...]


class CarYardSalesWorkflow:
    """A car sales assistant workflow with conversational memory."""

    def __init__(self, csv_path: str):
        self.state = SalesState()
        self.csv_path = Path(csv_path)
        self.query_engine = self._init_query_engine()

    def _init_query_engine(self):
        """Initialize CSV reader and query engine."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"❌ CSV file not found: {self.csv_path}")

        print(f"✅ Loading CSV data from: {self.csv_path}")
        reader = CSVReader()
        documents = reader.load_data(file=self.csv_path)

        print("✅ Building Vector Index...")
        index = VectorStoreIndex.from_documents(documents)

        print("✅ Creating Query Engine (OpenAI LLM)...")
        return index.as_query_engine(llm=OpenAI(model="gpt-4o-mini"))

    async def start(self, input_msg: str):
        """Start the workflow for one query."""
        self.state.query_count += 1
        self.state.last_question = input_msg
        print(f"\n[Step 1] Received Query #{self.state.query_count}: {input_msg}")
        return await self.process(input_msg)

    async def process(self, question: str):
        """Process query with LlamaIndex, including history."""
        print("[Step 2] Processing query via LlamaIndex...")

        # Add prior conversation context for continuity
        history_text = "\n".join(
            [f"User: {q}\nAgent: {a}" for q, a in self.state.chat_history]
        )
        full_query = (
            f"Conversation so far:\n{history_text}\n\n"
            f"New question: {question}"
        )

        try:
            response = await asyncio.to_thread(self.query_engine.query, full_query)
            answer = str(response)
        except Exception as e:
            answer = f"⚠️ Error: Could not process query. Details: {e}"

        return await self.end(question, answer)

    async def end(self, question: str, response: str):
        """Finalize the response and update memory."""
        print("[Step 3] Finalizing response.")
        self.state.last_answer = response
        self.state.chat_history.append((question, response))
        return response

    async def run(self, input_msg: str):
        """Run the full workflow pipeline."""
        return await self.start(input_msg)

    def show_chat_history(self):
        """Display the conversation so far."""
        print("\n🕮 Chat History:")
        for i, (q, a) in enumerate(self.state.chat_history, start=1):
            print(f"\n🧑 Q{i}: {q}\n🤖 A{i}: {a}")


async def main():
    # ✅ Update this to your actual CSV path
    csv_path = "/home/martin/Downloads/motorsales_bot_data - vehicles.csv"

    try:
        workflow = CarYardSalesWorkflow(csv_path)
    except FileNotFoundError as e:
        print(e)
        return

    # Sample dialogue
    questions = [
        "Show me all Toyotas under 1 million KES.",
        "Do you have any Subaru Forester?",
        "Which cars are most fuel efficient?",
        "Can you recommend one for a family of four?",
    ]

    for q in questions:
        response = await workflow.run(q)
        print("\n💬 Final Answer:", response)
        print("-" * 70)

    # ✅ View full chat history
    workflow.show_chat_history()


if __name__ == "__main__":
    asyncio.run(main())
