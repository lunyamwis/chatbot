import copy
import os
import re
import json
import uuid 
import pandas as pd
import requests
import logging
from openai import OpenAI
# import pandasai as pai
# from pandasai_litellm.litellm import LiteLLM

# # Initialize LiteLLM with your OpenAI model
# pandas_llm = LiteLLM(model="gpt-4.0-mini", api_key=os.getenv("OPENAI_API_KEY"))

# # Configure PandasAI to use this LLM
# pai.config.set({
#     "llm": pandas_llm
# })



llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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

FILTER_FIELDS = ["make", "model", "drive", "body_type", "colour", "budget", "stage"]
user_id = "martin"
conversation_memory[user_id] = {field: None for field in FILTER_FIELDS}
conversation_memory[user_id]["stage"] = "awaiting_model"
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





# -----------------------
# Tool: Filter cars
# -----------------------
def filter_cars_tool(make=None, model=None, drive=None, body_type=None, colour=None):
    """
    Filter the car dataset using memory and/or parameters.
    """
    filtered = df.copy()

    # Merge memory and explicit args
    memory = conversation_memory.get(user_id, {})
    criteria = memory.copy()
    if make: criteria["make"] = make
    if model: criteria["model"] = model
    if drive: criteria["drive"] = drive
    if body_type: criteria["body_type"] = body_type
    if colour: criteria["colour"] = colour

    for key, val in criteria.items():
        if val:
            col = normalize_colname(filtered, key)
            if col:
                filtered = filtered[filtered[col].astype(str).str.lower() == str(val).lower()]
    logging.info(f"[Tool] Filtered {len(filtered)} cars using memory: {criteria}")
    conversation_memory[user_id].update(criteria)
    logging.info(f"[Tool] Updated memory: {conversation_memory[user_id]}")
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
def call_openai_to_json(prompt, memory=None, filter_func=filter_cars_tool):
    """
    Call the LLM to extract memory updates, next stage, and generate a reply.
    Integrates filtering at every stage using a tool.
    """
    logging.info("Calling OpenAI API...")

    messages = [
        {"role": "system", "content": "You are a car sales assistant. Only return JSON, no text."},
        {"role": "user", "content": prompt}
    ]

    try:
        # --- Call the LLM with tools ---
        response = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "filter_cars_",
                        "description": "Filter the car dataset using known parameters (make, model, drive, etc.)",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "memory": {"type": "object"},
                                "make": {"type": "string"},
                                "model": {"type": "string"},
                                "drive": {"type": "string"},
                                "body_type": {"type": "string"},
                                "colour": {"type": "string"}
                            },
                            "required": ["memory"]
                        },
                    },
                }
            ],
            tool_choice="auto",
            response_format={"type": "json_object"}
        )

        # import pdb;pdb.set_trace()
        data = response.choices[0]
        content = data.message.content
        logging.info(f"Raw API response: {content}")
        # import pdb;pdb.set_trace()
        parsed = json.loads(content) if content else {}
        if content:
            parsed['reply'] = parsed.get('reply','') + parsed.get('next_question','')

        # --- integrate filtering at every stage ---
        if filter_func and memory:
            # Update memory with model’s extracted info
            # memory.update(parsed.get("memory_update", {}))

            # Handle tool calls
            if hasattr(data.message, "tool_calls") and data.message.tool_calls:
                tool_call = data.message.tool_calls[0]
                if tool_call.function.name == "filter_cars_":
                    args = json.loads(tool_call.function.arguments)
                    # import pdb;pdb.set_trace()
                    matches = filter_func(**args)

                    # Convert to DataFrame for easy grouping/filtering
                    logging.info(f"Tool returned {len(matches)} matches.")

                    # --- Stage-specific grouping ---
                    stage_to_fields = {
                        "awaiting_model": ["model","drive"],
                        "awaiting_drive": ["model", "drive", "body_type"],
                        "awaiting_body_type": ["model", "drive", "body_type", "colour"],
                        "awaiting_color": ["model", "drive", "body_type", "colour", "price"],
                        "awaiting_budget": ["model", "drive", "body_type", "colour", "price"],
                    }

                    current_stage = memory.get("stage", "awaiting_model")
                    # import pdb;pdb.set_trace()
                    # memory['stage'] = current_stage
                    current_fields = stage_to_fields.get(current_stage, ["model"])
                    group_field = current_fields[-1]
                    group_col = normalize_colname(matches, group_field) if not matches.empty else None

                    reply_lines = []
                    if group_col and group_col in matches.columns:
                        grouped = matches.groupby(group_col)
                        for group_value, group_df in grouped:
                            if pd.isna(group_value):
                                continue
                            reply_lines.append(f"\n{group_col.title()}: {group_value}")
                    else:
                        for _, row in matches.head(5).iterrows():
                            visible = ", ".join(
                                f"{c}: {row[normalize_colname(matches, c)]}"
                                for c in current_fields
                                if normalize_colname(matches, c) in matches.columns and pd.notna(row[normalize_colname(matches, c)])
                            )
                            reply_lines.append(visible)

                    # --- Final natural reply with matches ---
                    parsed["results"] = matches.to_dict(orient="records")
                    # paidf = pai.DataFrame(matches)
                    next_question_prompt = f"""
                    You are assisting in an interactive car sales conversation.
                    The goal is to guide the user step-by-step through the following stages:
                        awaiting_model → awaiting_drive → awaiting_body_type → awaiting_color → awaiting_budget

                    You must determine:
                    1. The most appropriate **next stage** in this sequence, given the user's current progress and the filtered car results.
                    2. The most relevant **next question** to ask that will help move to that stage.

                    Context:
                    - The already extracted information are summarized below:
                    {conversation_memory[user_id]}

                    Rules:
                    - Always follow the order of progression strictly.
                    - Do not skip stages, unless all information for earlier stages is already known.
                    - The output must be a pure JSON object in this exact format:
                    {{
                        "next_question": "string",
                        "next_stage": "string"
                    }}
                    """

                    next_question_response = llm.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a reasoning engine for a car sales assistant. "
                                    "Your job is to determine the next stage and next question based on car filtering results. "
                                    "Respond with JSON only, strictly formatted as {\"next_question\": \"...\", \"next_stage\": \"...\"}."
                                )
                            },
                            {"role": "user", "content": next_question_prompt}
                        ],
                        response_format={"type": "json_object"}
                    )

                    next_question = next_question_response.choices[0].message.content
                    parsed["next_question"] = json.loads(next_question).get("next_question", "")
                    parsed["next_stage"] = json.loads(next_question).get("next_stage", "")
                    conversation_memory[user_id]["stage"] = parsed["next_stage"]
                    print(f"Next question and stage: {parsed['next_question']} | {parsed['next_stage']}")

                    # memory["stage"] = parsed["next_stage"]
                    parsed["reply"] = parsed.get("next_question", "") + "\n" + parsed.get("reply", "Here are some options:") + "\n" + "\n".join(reply_lines)


        
        return parsed

    except Exception as e:
        logging.error(f"Error calling OpenAI or parsing response: {e}")
        return {
            "next_stage": "awaiting_model",
            "memory_update": {},
            "reply": "Sorry, could you clarify?"
        }




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

def classify_intent(conversation_history: list, customer_question: str, llm):
    """
    Categorize the user intent based on their message.
    Returns one of ['broker', 'window_shopper', 'buyer', 'unclear'].
    """
    prompt = f"""
    The following message is from a user inquiring about vehicles. 
    Categorize the user as one of based on the conversation history and their current question:
    - broker: they mention clients, commissions, reselling, multiple cars, or ask about trade terms
    - window_shopper: they are casually asking about prices, types, or just exploring
    - buyer: they show serious intent to purchase (asking availability, payment, condition)
    - unclear: cannot determine intent

    Conversation History: {json.dumps(conversation_history)}
    Current Question: "{customer_question}"

    Respond ONLY with valid JSON like:
    {{"intent": "<one of broker, window_shopper, buyer, unclear>"}}
    """

    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise intent classifier."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )

    # Extract the structured output
    return response.choices[0].message.content


def score_query_for_handoff(message: str, conversation_history: list, intent: str) -> float:
    score = 0.0
    text = message.lower().strip()
    intent  = json.loads(intent).get("intent", "unclear")
    print(intent)

    # existing emotion / human-handling signals
    if detect_robot_claim(text): score += 0.3
    if "talk to" in text and "person" in text: score += 0.6
    if len(text.split()) < 3: score += 0.1

    # intent-based scoring
    if intent == "broker":
        # likely not ideal for auto answers, prefer human
        score += 0.6
    elif intent == "window_shopper":
        # okay for bot, just mild caution
        score += 0.2
    elif intent == "buyer":
        # keep bot engaged, less handoff
        score -= 0.1

    return max(0.0, min(score, 1.0))


def FilterCarsTool(df):
    def _filter_cars(memory: dict) -> pd.DataFrame:
        filtered = df.copy()
        for key in ["make", "model", "drive", "body_type", "colour"]:
            val = memory.get(key)
            if val:
                col = normalize_colname(filtered, key)
                if col:
                    filtered = filtered[filtered[col].astype(str).str.lower() == str(val).lower()]
        logging.info(f"[FilterCarsTool] {len(filtered)} results after filtering by {memory}")
        return filtered.to_dict(orient="records")
    return _filter_cars


filter_cars_ = FilterCarsTool(df)



def generate_answer_v1(user_id, message):
    logging.info(f"User ({user_id}) says: {message}")
    
    if user_id not in conversation_memory:
        conversation_memory[user_id]["stage"] = "awaiting_model"

    memory = conversation_memory[user_id]
    logging.info(f"Current memory before processing: {memory}")

    # -----------------------
    # If awaiting budget
    # -----------------------
    if memory["stage"] == "completed":
        budget = extract_budget(message)
        if budget is None:
            logging.warning("Failed to parse budget from message.")
            return "I couldn't parse your budget. Please enter it like '2,100,000' or '2.1m'."
        memory["budget"] = budget

        # Filter cars based on memory and budget
        matches = filter_cars_(df)
        matches = matches.copy()
        price_col = normalize_colname(matches, "price")
        matches[price_col] = pd.to_numeric(matches[price_col], errors="coerce")
        matches = matches.dropna(subset=[price_col])
        matches["diff"] = (matches[price_col] - budget).abs()
        matches = matches.sort_values("diff").head(5)

        if matches.empty:
            logging.info("No cars match the budget.")
            return f"Sorry, none of the {memory.get('model','')} options fit your budget of {budget:,} KES."

        # Build reply with relevant fields for budget stage
        relevant_cols = ["model", "drive", "body_type", "colour", "price"]
        relevant_cols = [c for c in relevant_cols if normalize_colname(matches, c)]
        reply_lines = []
        for _, row in matches.iterrows():
            line = ", ".join(f"{c}: {row[normalize_colname(matches, c)]}" for c in relevant_cols)
            reply_lines.append(line)
        memory["stage"] = "completed"
        reply = "Here are some options matching your budget:\n" + "\n".join(reply_lines)
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
    5. Do not give examples or explanations.
    6. Ask the next user question based on the conversation memory given in order to reach "completed" stage.
    7. Ensure next question and generated reply do not clash or repeat.

    **Return ONLY valid JSON with this structure:**
    {{
        "next_stage": "...",
        "memory_update": {{}},
        "reply": "...",
        "next_question": "..."
    }}
    """
    parsed = call_openai_to_json(prompt, memory)
    print(f"Parsed LLM response: {parsed}")
    # update_memory(memory, parsed.get("memory_update", {}))
    # memory["stage"] = parsed.get("next_stage", memory["stage"])
    logging.info(f"Updated memory: {memory}")

    # -----------------------
    # Filter top 5 cars for the current stage
    # -----------------------
    
    return parsed.get("reply", "Could you please clarify?")





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
        Karimi: “Sure! The 2018 Prado in Pearl colour is going for 2.1 million.”

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
    intent = classify_intent(conversation_history, customer_question, llm)
    print(f"Classified intent: {intent}")
    score = score_query_for_handoff(customer_question, conversation_history, intent)

    logging.info(f"Intent: {intent} | Handoff score: {score}")

    if score >= 0.6:
        return {
            "escalation": True,
            "intent": intent,
            "score": score,
            "message": "I'll forward this to our human team — they’ll get back shortly."
        }
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
        # answer = handle_user_message(customer_question, conversation_memory)
        answer = generate_answer_v1(user_id, customer_question)
    except Exception as e:
        answer = "Just a sec have to urgently take care of something."
        print(f"Error occurred: {e}")

    print(f"Generated answer: {answer}")
    conversation_history.append({"user": customer_question, "assistant": answer})
    return answer

    # header = {
    #     "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY').strip()}",
    #     "Content-Type": "application/json",
    # }

    # body = {
    #     "model": "gpt-4o-mini",
    #     "messages": [
    #         {"role": "system", "content": rephrase_answer_prompt(customer_question, answer, conversation_history)},
    #         {"role": "user", "content": customer_question},
    #     ],
    # }

    # res = requests.post("https://api.openai.com/v1/chat/completions", json=body, headers=header)
    # gpt_response = ""
    # try:
    #     gpt_response = res.json()["choices"][0]["message"]["content"]
    # except Exception as e:
    #     gpt_response = "Be right back have to step out for a sec."
    #     print(f"Error occurred while saving response: {e}")

    # conversation_history.append({"user": customer_question, "assistant": gpt_response})
    # return gpt_response



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
