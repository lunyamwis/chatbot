import copy
import os
import re
import json
import uuid 
import pandas as pd
import requests
import logging

# -----------------------
# Logging setup
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------
# Load CSV
# -----------------------
df = pd.read_csv("/home/martin/Downloads/motorvehicles.csv")

# -----------------------
# Conversation memory
# -----------------------
conversation_memory = {}
conversation_history = []

FILTER_FIELDS = ["make", "model", "drive", "body_type", "color", "budget", "stage"]
user_id = "martin"
conversation_memory[user_id] = {field: None for field in FILTER_FIELDS}
# -----------------------
# Helpers
# -----------------------
def extract_budget(text: str):
    text = text.lower().replace(",", "").strip()
    m = re.search(r"(\d+(\.\d+)?)\s*(m|million)", text)
    if m:
        return int(float(m.group(1)) * 1_000_000)
    n = re.search(r"(\d{4,})", text)
    if n:
        return int(n.group(1))
    return None

def normalize_colname(df, col):
    for c in df.columns:
        if c.lower() == col.lower():
            return c
    return None

def filter_cars(df, memory):
    filtered = df.copy()
    for key in ["make","model","drive","body_type","color"]:
        val = memory.get(key)
        if val:
            col = normalize_colname(filtered, key)
            if col:
                filtered = filtered[filtered[col].astype(str).str.lower() == str(val).lower()]
    logging.info(f"Filtered cars ({len(filtered)} rows) based on memory: {memory}")
    return filtered

def pick_best_by_budget(df, budget):
    price_col = normalize_colname(df, "price")
    if price_col is None:
        logging.warning("Price column not found in CSV.")
        return None
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])
    if df.empty:
        logging.info("No cars have valid prices after conversion.")
        return None
    df["diff"] = (df[price_col] - budget).abs()
    best_row = df.sort_values("diff").head(1)  # DataFrame
    logging.info(f"Best match based on budget {budget}: {best_row.to_dict(orient='records')[0]}")
    return best_row

# -----------------------
# Call OpenAI API for JSON
# -----------------------
def call_openai_to_json(prompt):
    logging.info("Calling OpenAI API...")
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a car sales assistant. Only return JSON, no text."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }
    res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
    data = res.json()
    content = data["choices"][0]["message"]["content"]
    logging.info(f"Raw API response: {content}")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logging.error("JSON decoding failed.")
        return {"next_stage": "awaiting_model", "memory_update": {}, "reply": "Sorry, could you clarify?"}

# -----------------------
# Main conversation handler
# -----------------------
def update_memory(memory, memory_update):
    """
    Update conversation memory without overwriting existing values with None.
    """
    for key, value in memory_update.items():
        if value is not None:  # only update if new value is not None
            memory[key] = value
    return memory



def generate_answer(user_id, message):
    logging.info(f"User ({user_id}) says: {message}")
    if user_id not in conversation_memory:
        conversation_memory[user_id]["stage"] = "awaiting_model"

    memory = conversation_memory[user_id]
    
    
    
    logging.info(f"Current memory before processing: {memory}")

    # -----------------------
    # If awaiting budget
    # -----------------------
    if memory["stage"] == "awaiting_budget":
        budget = extract_budget(message)
        if budget is None:
            logging.warning("Failed to parse budget from message.")
            return "I couldn't parse your budget. Please enter it like '2,100,000' or '2.1m'."
        memory["budget"] = budget

        # Filter cars and pick best match
        matches = filter_cars(df, memory)
        best = pick_best_by_budget(matches, budget)
        if best is None or best.empty:
            logging.info("No cars match the budget.")
            return f"Sorry, none of the {memory.get('model','')} options fit your budget of {budget:,} KES."

        # Extract values safely
        model_col = normalize_colname(best, "model") or "model"
        color_col = normalize_colname(best, "colour") or "colour"
        price_col = normalize_colname(best, "price") or "price"
        mileage_col = normalize_colname(best, "mileage") or "mileage"

        reply = (f"Best match: {best[model_col].values[0]} ({best[color_col].values[0]}), "
                 f"Price: {int(best[price_col].values[0]):,} KES, "
                 f"Mileage: {best[mileage_col].values[0]} km.")

        memory["stage"] = "completed"
        logging.info(f"Memory after budget stage: {memory}")
        logging.info(f"Bot reply: {reply}")
        return reply

    # -----------------------
    # Otherwise, ask LLM to detect info and next stage
    # -----------------------
    prompt = f"""
        Conversation memory: {json.dumps(memory)}
        Customer message: "{message}"

        ### TASK
        1. Detect if the user is providing information for any of these fields: make, model, drive, body_type, color, budget
        2. Suggest the next stage of conversation: one of ["awaiting_model","awaiting_drive","awaiting_body_type","awaiting_color","awaiting_budget","completed"]
        3. Update memory with any new info
        4. Generate a short, natural reply to continue narrowing

        **Return ONLY valid JSON with this structure:**
        {{
            "next_stage": "...",
            "memory_update": {{}},
            "reply": "..."
        }}
        """

    parsed = call_openai_to_json(prompt)
    memory.update(parsed.get("memory_update", {}))
    memory["stage"] = parsed.get("next_stage", memory["stage"])
    logging.info(f"Updated memory: {memory}")
    logging.info(f"Bot reply: {parsed.get('reply')}")
    return parsed.get("reply","Sorry, could you clarify?")



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

def detect_robot_claim(text: str) -> bool:
    patterns = [
        r"\byou (are|r) (a )?robot\b",
        r"\byou (are|r) (a )?bot\b",
        r"\bnot human\b",
        r"\bjust a program\b",
        r"\bmachine learning\b",
        r"\bAI assistant\b",
        r"\bchatbot\b",
        r"\bnot real\b",
        r"\bfake\b",
        r"\bautomated response\b",
        r"\bscripted\b",
        r"\brespons(e|es) like a robot\b",
    ]
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True
    return False

user_id = "martin"
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
        answer = generate_answer(user_id, customer_question)
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
