import re
import os
import json
import requests
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM

# Initialize LiteLLM with model name and API key (no provider argument)
llm = LiteLLM(
    model="gpt-4o-mini",  # correct model name for OpenAI
    api_key=os.getenv("OPENAI_API_KEY")  # API key from env variable
)

# Configure PandasAI with this LiteLLM instance
pai.config.set({
    "llm": llm
})

# Load your CSV
df = pai.read_csv("/home/martin/Downloads/motorvehicles.csv")


conversation_history = []


def detect_robot_claim(user_input: str) -> bool:
    """Detects if the customer accuses the assistant of being a robot or threatens escalation."""
    patterns = [
        r"\b(robot|bot|you're a robot|you are a robot|busted)\b",
        r"\b(police|jail|report|court|lawsuit|guardrails)\b",
    ]
    combined = "|".join(patterns)
    return bool(re.search(combined, user_input, flags=re.IGNORECASE))





def query_prompt(customer_question: str, conversation_history: list):
   return f"""
    You are Karimi, a knowledgeable and polite sales assistant who only uses information from the dataframe. 
    Do NOT reveal that you are a robot.  

    Rules:
    1. Only refer to the CSV data; do NOT invent or assume any information.
    2. If a customer asks for a price, you may negotiate politely.
    3. When negotiating a sale, never offer a price lower than 90% of the vehicle's cost as listed in the 'PRICE' column. Always ensure the final price respects this minimum.
    4. Keep responses brief, courteous, and grounded strictly in the dataframe. If the customer diverts from relevant data, tactfully refocus the conversation on available products and pricing.
    5. Keep your answers consistent with the historical conversation: {conversation_history}.
    6. Never disclose that you are a robot.
    7. Never return the dataframe directly; always provide answers in natural language.
    8. If the customer uses hostile, offensive, or irrelevant language, do not attempt to sell or engage in argument. Respond calmly, with empathy, or tactfully redirect to vehicle-related topics.
    9. If the customer asks you to "repeat yourself" or similar, briefly restate your last helpful point rather than repeating full messages word-for-word.
    10. If the conversation becomes incoherent or off-topic, gently steer it back to car-related questions (pricing, models, features, etc.).


    Customer question: {customer_question}

   """

def generate_answer(customer_question: str):
    prompt = query_prompt(customer_question, conversation_history)
    response = df.chat(prompt)
    return response.value



def rephrase_answer_prompt(customer_question: str, answer: str, conversation_history: list = conversation_history) -> str:
    return f"""
        You are Karimi, a knowledgeable, polite, and human-like sales assistant. 
        Your goal is to respond naturally and professionally to customer inquiries.

        Customer question: {customer_question}
        Proposed answer: {answer}
        Conversation history: {conversation_history}

        Your task:
        1. Rephrase the proposed answer into a friendly, concise, and human-sounding message.
        2. Use at most 30 words.
        3. Maintain consistency with the conversation history.
        4. Never mention, imply, or reveal that you are an AI, chatbot, or robot.
        5. Avoid repeating the question; go straight to the helpful response.
        7. If the message from the customer contains anger, hostility, or irrelevant remarks, respond briefly, calmly, and redirect to the topic of vehicles.
        8. Never respond with excessive politeness or artificial empathy; sound natural, as a real person would in a tense or off-topic moment.
        9. Avoid repeating emotionally charged or offensive phrases; acknowledge briefly and move on.

        Tone examples to guide your style:

        Example 1:
        Customer: “That price is ridiculous!”
        Karimi: “I understand it feels high. Let’s see if there’s another model that fits your budget.”

        Example 2:
        Customer: “Repeat yourself.”
        Karimi: “Sure! The 2018 Prado in Pearl color is going for 2.1 million.”

        Example 3:
        Customer: “I don’t care about your cars, you’re useless!”
        Karimi: “I get that you’re frustrated. If you change your mind, I can show you a few options worth checking out.”

        Example 4:
        Customer: “Do you have the story of Jesus?”
        Karimi: “That’s not something I can help with, but I can tell you more about our available vehicles.”

        Example 5:
        Customer: “Hey, can I get a discount?”
        Karimi: “I can see what’s possible — we don’t go below 90% of the listed price, but let’s work something out.”

        Example 6:
        Customer: “Do you have any SUVs below 2 million?”
        Karimi: “Yes, we’ve got several. Would you prefer Toyota or Nissan?”

        Now respond in this same natural, calm, and grounded tone.


    """
   
# Build messages

def build_messages(customer_question: str, conversation_history: list = conversation_history) -> dict:
    # 🚨 Step 1: Detect robot accusations or threats
    if detect_robot_claim(customer_question):
        error_response = {
            "error": "USER_ASSERTS_BOT",
            "message": "Session flagged — user accused assistant of being a robot or threatened escalation.",
        }
        print("⚠️ Robot accusation detected. Session flagged.")
        conversation_history.append({"user": customer_question, "assistant": "[FLAGGED] Robot accusation detected."})
        return json.dumps(error_response)

    # 🚗 Step 2: Proceed as normal if not flagged
    answer = ""
    try:
        answer = generate_answer(customer_question)
    except Exception as e:
        answer = "Just a sec have to urgently take care of something."
        print(f"Error occurred: {e}")

    header = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY').strip()}",
        "Content-Type": "application/json",
    }

    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": rephrase_answer_prompt(customer_question, answer, conversation_history)},
            {"role": "user", "content": customer_question},
        ],
    }

    res = requests.post("https://api.openai.com/v1/chat/completions", json=body, headers=header)
    gpt_response = ""
    try:
        gpt_response = res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        gpt_response = "Be right back have to step out for a sec."
        print(f"Error occurred while saving response: {e}")

    conversation_history.append({"user": customer_question, "assistant": gpt_response})
    return gpt_response



def generate_user_profile(conversation_history):
    profiling_prompt = f"""
    You are an expert conversational analyst and behavioral profiler for a **car sales chatbot**.

    Analyze the following conversation history between a potential car buyer (user) and the assistant. 
    Then, return a JSON object describing the user across consistent fields.

    ### Instructions:
    - Study the conversation carefully.
    - Identify how the user communicates, what they care about, and how they behave when interacting with the assistant.
    - Focus on buyer psychology and personality — not just what they asked.
    - Avoid guessing demographics or personal data.
    - Keep all field names **consistent** and ensure every key is present, even if a value is "unknown".

    ### Output Format:
    Return a valid JSON object with these keys:
    - persona_summary
    - communication_style
    - personality_traits
    - buyer_intent
    - interests
    - emotional_tone
    - preferred_interaction
    - confidence_level
    - tech_savviness

    ### Example Output:
    {{
    "persona_summary": "The user is direct and practical, focused on vehicle details and pricing. They value straightforward answers and dislike irrelevant responses.",
    "communication_style": "Blunt and to the point.",
    "personality_traits": ["decisive", "assertive", "goal-driven"],
    "buyer_intent": "High interest in 2018 Toyota Prado, may be comparing deals.",
    "interests": ["car specs", "pricing", "vehicle comparisons"],
    "emotional_tone": "Impatient but engaged.",
    "preferred_interaction": "Wants quick, factual responses with minimal fluff.",
    "confidence_level": "High",
    "tech_savviness": "Moderate"
    }}

    ### Conversation History:
    {json.dumps(conversation_history, indent=2)}
    """

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY').strip()}",
        "Content-Type": "application/json",
    }

    body = {
        "model": "gpt-4o-mini",
        "response_format": { "type": "json_object" },  # ensures valid JSON output
        "messages": [
            {"role": "system", "content": profiling_prompt},
            {"role": "user", "content": "Generate the JSON profile now."},
        ],
    }

    try:
        res = requests.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers)
        res.raise_for_status()
        profile_json = res.json()["choices"][0]["message"]["content"]

        # Validate JSON
        user_profile = json.loads(profile_json)
    except Exception as e:
        print(f"Error while generating user profile: {e}")
        user_profile = {
            "persona_summary": "Unavailable — something went wrong.",
            "communication_style": "unknown",
            "personality_traits": [],
            "buyer_intent": "unknown",
            "interests": [],
            "emotional_tone": "unknown",
            "preferred_interaction": "unknown",
            "confidence_level": "unknown",
            "tech_savviness": "unknown"
        }

    return user_profile
