import os
import re
import json
import uuid
import logging
import sqlite3
import pandas as pd
import requests

# -----------------------
# Logging setup
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------
# Load CSV
# -----------------------
df = pd.read_csv("/home/martin/Downloads/motorvehicles.csv")

# -----------------------
# SQLite memory setup
# -----------------------
DB_PATH = "chat_memory.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS user_memory (
    user_id TEXT PRIMARY KEY,
    memory TEXT
)
""")
conn.commit()

FILTER_FIELDS = ["make", "model", "drive", "body_type", "color", "budget"]

# -----------------------
# Helper functions
# -----------------------

user_id = str(uuid.uuid4())

def log_message(user_id, role, message):
    timestamp = int(pd.Timestamp.now().timestamp())
    c.execute("CREATE TABLE IF NOT EXISTS history (user_id TEXT, role TEXT, message TEXT, timestamp INTEGER)")
    c.execute("INSERT INTO history (user_id, role, message, timestamp) VALUES (?, ?, ?, ?)", (user_id, role, message, timestamp))
    conn.commit()
    logging.info(f"Logged message for {user_id}: [{role}] {message}")

def save_memory(user_id, memory):
    serialized = json.dumps(memory)
    c.execute("INSERT OR REPLACE INTO user_memory (user_id, memory) VALUES (?, ?)", (user_id, serialized))
    conn.commit()
    logging.info(f"Memory saved for {user_id}: {memory}")

def load_memory(user_id):
    c.execute("SELECT memory FROM user_memory WHERE user_id=?", (user_id,))
    row = c.fetchone()
    if row:
        return json.loads(row[0])
    return {"stage": "awaiting_model", **{f: None for f in FILTER_FIELDS}}

def extract_budget(text):
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
    best_row = df.sort_values("diff").head(1)
    logging.info(f"Best match based on budget {budget}: {best_row.to_dict(orient='records')[0]}")
    return best_row

def detect_robot_claim(text):
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

# -----------------------
# Call OpenAI for JSON
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
# Generate answer
# -----------------------
def generate_answer(user_id, message):
    logging.info(f"User ({user_id}) says: {message}")
    if detect_robot_claim(message):
        return "⚠️ I understand your concern, let's continue with vehicle selection."

    memory = load_memory(user_id)
    logging.info(f"Loaded memory: {memory}")

    # Budget stage
    if memory.get("stage") == "awaiting_budget":
        budget = extract_budget(message)
        if budget is None:
            return "I couldn't parse your budget. Please enter it like '2,100,000' or '2.1m'."
        memory["budget"] = budget
        matches = filter_cars(df, memory)
        best = pick_best_by_budget(matches, budget)
        if best is None or best.empty:
            save_memory(user_id, memory)
            return f"Sorry, none of the {memory.get('model','')} options fit your budget of {budget:,} KES."
        # Safe extraction
        model_col = normalize_colname(best,"model") or "model"
        color_col = normalize_colname(best,"color") or "color"
        price_col = normalize_colname(best,"price") or "price"
        mileage_col = normalize_colname(best,"mileage") or "mileage"
        reply = (f"Best match: {best[model_col].values[0]} ({best[color_col].values[0]}), "
                 f"Price: {int(best[price_col].values[0]):,} KES, "
                 f"Mileage: {best[mileage_col].values[0]} km.")
        memory["stage"] = "completed"
        save_memory(user_id, memory)
        logging.info(f"Memory after budget: {memory}")
        return reply

    # Otherwise, ask LLM for next stage
    prompt = f"""
Conversation memory: {json.dumps(memory)}
Customer message: "{message}"

### TASK
1. Detect if the user provides info for these fields: make, model, drive, body_type, color, budget
2. Suggest next stage: ["awaiting_model","awaiting_drive","awaiting_body_type","awaiting_color","awaiting_budget","completed"]
3. Update memory with new info
4. Generate a short natural reply

Return ONLY JSON: {{"next_stage":"...","memory_update":{{}},"reply":"..."}}
"""
    parsed = call_openai_to_json(prompt)
    memory.update(parsed.get("memory_update", {}))
    memory["stage"] = parsed.get("next_stage", memory.get("stage"))
    save_memory(user_id, memory)
    logging.info(f"Updated memory: {memory}")
    logging.info(f"Bot reply: {parsed.get('reply')}")
    return parsed.get("reply","Sorry, could you clarify?")

# -----------------------
# Build messages and rephrase
# -----------------------
conversation_history = []

def rephrase_answer_prompt(customer_question: str, answer: str, conversation_history: list = conversation_history) -> str:
    return f"""
You are Karimi, a knowledgeable car sales assistant. Rephrase the proposed answer naturally.

Customer question: {customer_question}
Proposed answer: {answer}
Conversation history: {conversation_history}

Respond friendly, concise, max 30 words. Ignore off-topic or angry remarks.
"""

def build_messages(customer_question: str) -> str:
    if detect_robot_claim(customer_question):
        log_message(user_id, "user", customer_question)
        log_message(user_id, "assistant", "[FLAGGED] Robot accusation detected.")
        print("⚠️ Robot accusation detected. Session flagged.")
        return "[FLAGGED] Robot accusation detected."

    answer = generate_answer(user_id, customer_question)

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

    try:
        res = requests.post("https://api.openai.com/v1/chat/completions", json=body, headers=header)
        gpt_response = res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        gpt_response = answer
        logging.error(f"Error while rephrasing: {e}")

    conversation_history.append({"user": customer_question, "assistant": gpt_response})
    log_message(user_id, "assistant", gpt_response)
    return gpt_response

# -----------------------
# User profiling
# -----------------------


def generate_user_profile(user_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role,message FROM history WHERE user_id=? ORDER BY timestamp", (user_id,))
    history_rows = c.fetchall()
    conversation_history = [{"role": r, "message": m} for r,m in history_rows]
    conn.close()

    profiling_prompt = f"""
Analyze this conversation with a car sales assistant. Return JSON with:
persona_summary, communication_style, personality_traits, buyer_intent, interests, emotional_tone, preferred_interaction, confidence_level, tech_savviness.

Conversation history: {json.dumps(conversation_history)}
"""
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY').strip()}",
        "Content-Type": "application/json",
    }

    body = {
        "model": "gpt-4o-mini",
        "response_format": { "type": "json_object" },
        "messages": [
            {"role": "system", "content": profiling_prompt},
            {"role": "user", "content": "Generate the JSON profile now."},
        ],
    }

    try:
        res = requests.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers)
        profile_json = res.json()["choices"][0]["message"]["content"]
        user_profile = json.loads(profile_json)
    except Exception as e:
        logging.error(f"Error generating profile: {e}")
        user_profile = {k: "unknown" for k in ["persona_summary","communication_style","personality_traits","buyer_intent","interests","emotional_tone","preferred_interaction","confidence_level","tech_savviness"]}

    return user_profile