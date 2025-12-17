import pandas as pd
import os
import requests
# Load CSV
df = pd.read_csv("/home/martin/Downloads/motorvehicles.csv")

# Convert to text (you can also limit columns/rows if large)
csv_text = df.to_csv(index=False)
conversation_history = []

def generate_prompt(customer_question: str, answer: str, conversation_history: list = conversation_history) -> str:
    return f"""
    You are Karimi, a knowledgeable and polite sales assistant who matches customer questions to the answer, 
    Customer question: {customer_question}

    Answer: {answer}

    your work is to rephrase the answer in a polite and concise manner, do NOT reveal that you are a robot.
    1.use only 30 words maximum.
    2.Keep your answers consistent with the historical conversation: {conversation_history}.
    3.Never disclose that you are a robot.


    """
   
# Build messages

def build_messages(customer_question: str, answer: str, conversation_history: list = conversation_history) -> dict:
    header = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY').strip()}",
            "Content-Type": "application/json",
    }
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": generate_prompt(customer_question, answer, conversation_history)},
            {"role": "user", "content": customer_question}
        ]
    }
    res = requests.post("https://api.openai.com/v1/chat/completions", json=body, headers=header)
    # save the response to the database
    gpt_response = res.json()["choices"][0]["message"]["content"]
    conversation_history.append({"user": customer_question, "assistant": gpt_response})
    return gpt_response
