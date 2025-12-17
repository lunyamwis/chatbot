import pandasai as pai
from pandasai_litellm.litellm import LiteLLM
import pandas as pd
import os
import gradio as gr

# Load your car sales CSV
df = pai.read_csv("/home/martin/Downloads/botdata.csv")

# Initialize LiteLLM
llm = LiteLLM(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Configure PandasAI
pai.config.set({
    "llm": llm,
    "verbose": True,
    "system_prompt": f"""
You are a car sales assistant. Answer ONLY using this car sales data:
Columns: {df.columns.tolist()}

Examples:
"Cheapest SUV?" → Filter BODY TYPE='SUV', sort PRICE
"Red Toyotas 2023?" → MAKE='Toyota', COLOUR='Red', YEAR=2023
"Top 5 deals?" → Sort (PRICE/MILEAGE) ascending

Use pandas operations: filter, groupby, sort_values, describe.
"""
})

# Fixed chat function - handles PandasAI response format
def chat(message, history):
    result = df.chat(message)
    response = str(result.value)
    bot_response = f"You said: {response}"  # replace with your CSV agent call
    
    history.append((response, bot_response))
    return history, history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type your question here...")
    clear = gr.Button("Clear")

    msg.submit(chat, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch()
