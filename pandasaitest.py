import copy
import os
import re
import json
import uuid 
import pandas as pd
import requests
import logging
import math
from openai import OpenAI
# import pandasai as pai
# from pandasai_litellm.litellm import LiteLLM

# # Initialize LiteLLM with your OpenAI model
# pandas_llm = LiteLLM(model="gpt-4.0-mini", api_key=os.getenv("OPENAI_API_KEY"))

# # Configure PandasAI to use this LLM
# pai.config.set({
#     "llm": pandas_llm
# })

import sqlite3
from contextlib import closing

DB_PATH = "conversation_memory.db"



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

FILTER_FIELDS = ["make", "model", "drive", "body_type", "colour", "budget", "stage","next_stage"]
stages_mapper = {
    "awaiting_model": 0,
    "awaiting_drive": 1,
    "awaiting_body_type": 2,
    "awaiting_colour": 3,
    "awaiting_budget": 4,
    "completed": 5
}
stages_list = [
    "awaiting_model",
    "awaiting_drive",
    "awaiting_body_type",
    "awaiting_colour",
    "awaiting_budget",
    "completed"
]
user_id = "martin"
conversation_memory[user_id] = {field: None for field in FILTER_FIELDS}
conversation_memory[user_id]["stage"] = "awaiting_model"

def init_memory_db():
    """Initialize the SQLite table with real columns."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_memory (
                user_id TEXT PRIMARY KEY,
                make TEXT,
                model TEXT,
                drive TEXT,
                body_type TEXT,
                colour TEXT,
                budget INTEGER,
                stage TEXT
            )
        """)
        conn.commit()


def load_memory_from_db(user_id: str) -> dict:
    """Load the memory for a given user_id; return defaults if not found."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.execute("SELECT make, model, drive, body_type, colour, budget,stage FROM conversation_memory WHERE user_id = ?", (user_id,))
        row = cur.fetchone()

    if row:
        make, model, drive, body_type, colour, budget, stage = row
    else:
        make = model = drive = body_type = colour = None
        budget = 0
        stage = "awaiting_model"

    return {
        "make": make,
        "model": model,
        "drive": drive,
        "body_type": body_type,
        "colour": colour,
        "budget": budget,
        "stage": stage
    }

def save_memory_to_db(user_id: str, memory: dict):
    # import pdb;pdb.set_trace()
    """Insert or update the user's memory record."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""
            INSERT INTO conversation_memory (user_id, make, model, drive, body_type, colour, budget, stage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                make=excluded.make,
                model=excluded.model,
                drive=excluded.drive,
                body_type=excluded.body_type,
                colour=excluded.colour,
                budget=excluded.budget,
                stage=excluded.stage
        """, (
            user_id,
            memory.get("make"),
            memory.get("model"),
            memory.get("drive"),
            memory.get("body_type"),
            memory.get("colour"),
            memory.get("budget", 0),
            memory.get("stage", "awaiting_model"),
        ))
        conn.commit()

def clear_memory_in_db(user_id: str):
    """Remove a user’s memory record."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("DELETE FROM conversation_memory WHERE user_id = ?", (user_id,))
        conn.commit()

# -----------------------
# Helpers
# -----------------------
def extract_budget(text: str):
    text = text.lower().replace(",", "").strip()
    m = re.search(r"(\d+(\.\d+)?)\s*(m|million|k|thousand)", text)
    if m:
        return int(float(m.group(1)) * 1_000_000) if m.group(3) in ["m", "million"] else int(float(m.group(1)) * 1_000)
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
def filter_cars_tool(user_id=user_id, make=None, model=None, drive=None, body_type=None, colour=None):
    """
    Filter the car dataset using user parameters + saved memory.
    Falls back between 'make' and 'model' if needed.
    Persists updates in structured DB columns.
    """
    # import pdb;pdb.set_trace()
    filtered = df.copy()
    memory = load_memory_from_db(user_id)

    # Merge with new info
    if make: memory["make"] = make 
    if model: memory["model"] = model if model else make
    if drive: memory["drive"] = drive
    if body_type: memory["body_type"] = body_type
    if colour: memory["colour"] = colour

    # Apply filtering logic
    for key, val in memory.items():
        if not val or key == "stage":
            continue

        if key in ["make", "model"]:
            make_col = normalize_colname(filtered, "make")
            model_col = normalize_colname(filtered, "model")
            mask = pd.Series(False, index=filtered.index)
            if make_col in filtered.columns:
                mask |= filtered[make_col].astype(str).str.lower().str.contains(str(val).lower(), na=False)
            if model_col in filtered.columns:
                mask |= filtered[model_col].astype(str).str.lower().str.contains(str(val).lower(), na=False)
            filtered = filtered[mask]
        else:
            col = normalize_colname(filtered, key)
            if col in filtered.columns:
                filtered = filtered[filtered[col].astype(str).str.lower().str.contains(str(val).lower(), na=False)]

    # Persist the latest memory state
    if filtered.empty:
        logging.info(f"[Tool] No matches found when filtering with memory: {memory}")
    else:
        save_memory_to_db(user_id, memory)
    logging.info(f"[Tool] Filtered {len(filtered)} cars using structured memory: {memory}")
    return filtered


def filter_cars_tool_v1(user_id=None, make=None, model=None, drive=None, body_type=None, colour=None):
    """
    Filter the car dataset using memory and/or parameters.
    Performs fallback matching (e.g., if 'make' not found, checks in 'model').
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
        # conversation_memory[user_id][key] = val  # Update memory with latest criteria
        if not val:
            continue

        col = normalize_colname(filtered, key)

        # --- Fallback: check both make and model if relevant ---
        if key in ["make", "model"]:
            make_col = normalize_colname(filtered, "make")
            model_col = normalize_colname(filtered, "model")

            mask = pd.Series(False, index=filtered.index)
            if make_col in filtered.columns:
                mask |= filtered[make_col].astype(str).str.lower().str.contains(str(val).lower(), na=False)
            if model_col in filtered.columns:
                mask |= filtered[model_col].astype(str).str.lower().str.contains(str(val).lower(), na=False)

            filtered = filtered[mask]
        else:
            # Normal filtering for other attributes
            if col in filtered.columns:
                filtered = filtered[
                    filtered[col].astype(str).str.lower().str.contains(str(val).lower(), na=False)
                ]

    logging.info(f"[Tool] Filtered {len(filtered)} cars using memory: {criteria}")
    conversation_memory[user_id] = criteria
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

def update_stage(user_id, new_stage):
    memory = load_memory_from_db(user_id)
    memory["stage"] = new_stage
    save_memory_to_db(user_id, memory)
    logging.info(f"[Memory] Stage updated for {user_id}: {new_stage}")

def update_make_or_model(user_id, new_make=None, new_model=None):
    memory = load_memory_from_db(user_id)
    if new_make:
        memory["make"] = new_make
    if new_model:
        memory["model"] = new_model if new_model else new_make
    save_memory_to_db(user_id, memory)
    logging.info(f"[Memory] Make/Model updated for {user_id}: {new_make}, {new_model}")

def update_drive(user_id, new_drive):
    memory = load_memory_from_db(user_id)
    memory["drive"] = new_drive
    save_memory_to_db(user_id, memory)
    logging.info(f"[Memory] Drive updated for {user_id}: {new_drive}")


def update_body_type(user_id, new_body_type):
    memory = load_memory_from_db(user_id)
    memory["body_type"] = new_body_type
    save_memory_to_db(user_id, memory)
    logging.info(f"[Memory] Body type updated for {user_id}: {new_body_type}")

def update_colour(user_id, new_colour):
    memory = load_memory_from_db(user_id)
    memory["colour"] = new_colour
    save_memory_to_db(user_id, memory)
    logging.info(f"[Memory] Colour updated for {user_id}: {new_colour}")

def update_budget(user_id, new_budget):
    memory = load_memory_from_db(user_id)
    memory["budget"] = int(new_budget)
    save_memory_to_db(user_id, memory)
    logging.info(f"[Memory] Budget updated for {user_id}: {new_budget}")


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
        {"role": "system", "content": "You are a car sales assistant. Only return JSON, no text. You MUST always call the function `filter_cars_` whenever the user message involves vehicle details (make, model, drive, color, budget, etc.)."},
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
            # tool_choice={"type": "function", "function": {"name": "filter_cars_"}}, 
            response_format={"type": "json_object"}
        )

        # import pdb;pdb.set_trace()
        data = response.choices[0]
        print(data)
        content = data.message.content
        logging.info(f"Raw API response: {content}")
        # import pdb;pdb.set_trace()
        if content:
            parsed_content = json.loads(content)
            if "next_stage" in parsed_content:
                update_stage(user_id, new_stage=parsed_content["next_stage"])
            if "make" in parsed_content["memory_update"] or "model" in parsed_content["memory_update"]:
                if memory['make'] or memory['model']:
                    if memory.get("make") != parsed_content['memory_update'].get("make","") or memory.get("model") != parsed_content['memory_update'].get("model",""): # to cater for multiple searches
                        parsed_content['memory_update'] = {k: v for k, v in parsed_content['memory_update'].items() if k in ("make", "model")}
                        clear_memory_in_db(user_id)
                        print("Cleared memory due to new make/model search. Preparing for new search...")
                        load_memory_from_db(user_id) # refresh memory after clearing
                update_make_or_model(user_id, new_make=parsed_content['memory_update'].get("make", "") if parsed_content['memory_update'].get("make", "") else parsed_content['memory_update'].get("model", ""), new_model=parsed_content['memory_update'].get("model", "") if parsed_content['memory_update'].get("model", "") else parsed_content['memory_update'].get("make", ""))

            if "drive" in parsed_content["memory_update"]:
                update_drive(user_id, new_drive=parsed_content['memory_update'].get("drive", ""))
            if "body_type" in parsed_content["memory_update"]:
                update_body_type(user_id, new_body_type=parsed_content['memory_update'].get("body_type", ""))
            if "colour" in parsed_content["memory_update"]:
                update_colour(user_id, new_colour=parsed_content['memory_update'].get("colour", ""))
            if "budget" in parsed_content["memory_update"]:
                update_budget(user_id, new_budget=parsed_content['memory_update'].get("budget", ""))
        parsed = json.loads(content) if content else {}
        if content:
            if "memory_update" in parsed and filter_func:
                # reload memory
                memory = load_memory_from_db(user_id)
                filtered_data = {k: v for k, v in memory.items() if k != 'stage' and k != 'budget'}
                matches = filter_func(**filtered_data)
                reply = return_matches_as_text(parsed, matches, is_off_tool=True)
                parsed['reply'] =  reply 
            else:
                parsed['reply'] = parsed.get('reply','') + parsed.get('next_question','')

        # --- integrate filtering at every stage ---

        if filter_func and memory:
            # Update memory with model’s extracted info
            # memory.update(parsed.get("memory_update", {}))

            # Handle tool calls
            memory_update = parsed.get("memory_update", {})
            if hasattr(data.message, "tool_calls") and data.message.tool_calls:
                # reload memory
                memory = load_memory_from_db(user_id)
                # import pdb;pdb.set_trace()
                
                tool_call = data.message.tool_calls[0]
                if tool_call.function.name == "filter_cars_":
                    args = json.loads(tool_call.function.arguments)
                    # import pdb;pdb.set_trace()
                    if "make" in args or "model" in args:
                        if memory['make'] or memory['model']:
                            if memory.get("make") != args.get("make","") or memory.get("model") != args.get("model",""): # to cater for multiple searches
                                args = {k: v for k, v in args.items() if k in ("make", "model")}
                                clear_memory_in_db(user_id)
                                print("Cleared memory due to new make/model search. Preparing for new search...")
                                load_memory_from_db(user_id) # refresh memory after clearing

                    update_make_or_model(user_id, new_make=args.get("make", "") if args.get("make", "") else args.get("model", ""), new_model=args.get("model", "") if args.get("model", "") else args.get("make", ""))
                    filtered_dataset_ = {k: v for k, v in args.items() if k != 'stage' and k != 'budget'}
                    matches = filter_func(**filtered_dataset_)
                    # import pdb;pdb.set_trace()
                    reply = return_matches_as_text(parsed, matches,is_off_tool=False)
                    parsed["reply"] = reply
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

def return_matches_as_text(parsed, matches, is_off_tool=False) -> str:
    # import pdb;pdb.set_trace()
    if matches.empty:
        # Get all available models from your DataFrame
        model_col = normalize_colname(df, "model")
        available_models = df[model_col].dropna().unique().tolist()

        response_empty = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a friendly and helpful car sales assistant. "
                        f"keep into context the conversation_history: {conversation_history}"
                        "The user asked for cars, but no exact matches were found. "
                        f"Available models: {available_models}. "
                        "Suggest the closest matching options from this list. "
                        "Return your response strictly as a JSON object with these keys:\n"
                        "  reply: the message to display to the user\n"
                        "  suggested_models: a list of closest available models\n"
                        "  next_action: what the assistant should do next (e.g., 'ask_if_user_wants_suggestions')\n"
                        "Do not include any text outside the JSON object."
                    )
                },
                {
                    "role": "user",
                    "content": "No matching cars were found based on the current criteria."
                }
            ],
            response_format={"type": "json_object"}
        )

        # Parse the LLM response (depends on SDK, adjust if needed)
        llm_result = response_empty.choices[0].message.content
        # import pdb;pdb.set_trace()
        response_gotten = json.loads(llm_result).get("reply", "Sorry, I couldn't find any cars matching your criteria.") + ','.join(json.loads(llm_result).get("suggested_models", ""))
        return response_gotten  # JSON object: {"reply": "...", "suggested_models": [...], "next_action": "..."}

        

    # Convert to DataFrame for easy grouping/filtering
    logging.info(f"Tool returned {len(matches)} matches.")
    if len(matches) == 0:
        parsed["reply"] = "Sorry, I couldn't find any cars matching your criteria. Could you try other options?"
        return parsed
    

    # --- Stage-specific grouping ---
    stage_to_fields = {
        "awaiting_model": ["model"],
        "awaiting_drive": ["model", "drive"],
        "awaiting_body_type": ["model", "drive", "body_type"],
        "awaiting_colour": ["model", "drive", "body_type", "colour"],
        "awaiting_budget": ["model", "drive", "body_type", "colour", "price"],
    }

    # load_memory_from_db(user_id) again to refresh changes after filtering
    memory = load_memory_from_db(user_id)
    current_stage = memory.get("stage", "awaiting_model")
    # import pdb;pdb.set_trace()
    # memory['stage'] = current_stage
    stages_index = 0
    if not is_off_tool:
        if memory.get("model") or memory.get("make"):
            stages_index = stages_mapper[current_stage]+1
        else:
            stages_index = stages_mapper[current_stage]
    else:
        stages_index = stages_mapper[current_stage]
    current_fields = stage_to_fields.get(stages_list[stages_index], ["model"])
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
        awaiting_model → awaiting_drive → awaiting_body_type → awaiting_colour → awaiting_budget

    You must determine:
    1. The most appropriate **next stage** in this sequence, given the user's current progress and the filtered car results.
    2. The most relevant **next question** to ask that will help move to that stage.
    

    Context:
    - The already extracted information are summarized below:
    {load_memory_from_db(user_id)}
    

    Rules:
    - Always follow the order of progression strictly.
    - Do not skip stages, unless all information for earlier stages is already known.
    - The output must be a pure JSON object in this exact format:
    - Do not give any options or examples at all!!!!!!!!!!
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
    conversation_memory[user_id]["next_stage"] = parsed["next_stage"]
    # print(f"Saving memory: {memory}")
    if not is_off_tool:
        update_stage(user_id, parsed["next_stage"])
    
    print("Memory after stage update:", load_memory_from_db(user_id))
    print(f"Next question and stage: {parsed['next_question']} | {parsed['next_stage']}")

    # memory["stage"] = parsed["next_stage"]
    if parsed["next_stage"] == "awaiting_budget" and memory.get("colour") and memory.get("body_type") and memory.get("drive") and (memory.get("model") or memory.get("make")):
        parsed["reply"] = "What is your budget for the vehicle?"
    else:
        parsed["reply"] = parsed.get("next_question", "") + "\n" + parsed.get("reply", "Here are options from our stock:") + "\n" + "\n".join(reply_lines)

    return parsed["reply"]


def negotiate_price_from_memory(memory, df, user_reply_text):
    """
    memory: dict-like object that should include 'budget' (numeric) and optionally other info
    df: pandas DataFrame with available cars
    user_reply_text: the immediate user message (e.g., "I want this on hire purchase..." or an answer)
    Returns: final_result dict with structure:
      {
        "listed_price": <int>,
        "starting_offer": <int>,
        "final_offer": <int>,
        "closed": <bool>,
        "conversation_history": [ { "role": "assistant"/"user", "text": "...", "offer": <int or None> }, ... ],
        "memory_update": { ... }
      }
    """
    # --- Step 0: get budget from memory (fallback to user input if missing)
    budget = None
    if isinstance(memory.get("budget"), (int, float, str)):
        try:
            budget = float(memory.get("budget"))
        except Exception:
            budget = None


    # --- Step 1: pick best car row
    best_row_df = df
    geolocation_col = normalize_colname(df, "geolocation")
    best_geolocation = df[geolocation_col].iloc[0] if not df.empty else None
    price_col = normalize_colname(best_row_df, "selling_price")
    if best_row_df is None:
        return {"error": "No matching car or price column not found."}

    listed_price = int(round(float(best_row_df[price_col].iloc[0])))

    # Starting offer = 90% of selling price (i.e., already at 10% off)
    starting_offer = int(math.ceil(listed_price * 0.9))

    # Negotiation floor is 50% of selling price (we must not go below this unless exceptional)
    negotiation_floor = int(math.floor(listed_price * 0.5))

    # Prepare negotiation state
    current_offer = starting_offer
    closed = False
    # conversation_history.append({"user":user_reply_text, "assistant": f"Initial offer: KES {current_offer:,}"})

    # Load negotiation rules from file (optional)
    try:
        with open("negotiation_rules.txt", "r") as f:
            negotiation_rules = f.read()
    except FileNotFoundError:
        negotiation_rules = "No negotiation rules file found."

    # Safety: prevent infinite loops by limiting iterations
    max_rounds = 3

    # Each iteration we send the conversation + state to the LLM, and ask it to:
    # - respond to the user (reply)
    # - either accept the user's latest proposal, or produce a new counter-offer (next_offer)
    # - indicate if sale is closed (closed: true/false)
    # We enforce JSON-only response via response_format that returns a JSON object (SDK-dependent).
    for round_idx in range(max_rounds):
        # Build messages
        system_content = (
            "You are a car sales negotiation assistant. When negotiating, be polite and strategic. "
            "You must never offer below the negotiation_floor unless explicitly instructed. "
            "Start from the current_offer (which is 90% of the listed price). "
            "You should attempt step-by-step concessions, closing the deal if the buyer agrees. "
            "Each response MUST be a single JSON object only (no extra text) with keys:\n"
            "  next_stage (string),\n"
            "  memory_update (object),\n"
            "  reply (string) - assistant visible message,\n"
            "  next_offer (number or null),\n"
            "  closed (true/false)\n"
            "Do not include anything else outside the JSON object."
        )

        # Compose the user content to give the LLM full state
        user_content = (
            f"Negotiation rules:\n{negotiation_rules}\n\n"
            f"Memory:\n{json.dumps(memory, indent=2)}\n\n"
            f"Car listed price: {listed_price}\n"
            f"Current offer: {current_offer}\n"
            f"Negotiation floor (do not go below): {negotiation_floor}\n"
            f"Buyer budget: {budget}\n\n"
            f"conversation_history so far:\n{json.dumps(conversation_history, indent=2)}\n\n"
            f"Buyer says: \"{user_reply_text}\"\n\n"
            "Based on the buyer's message, decide whether to:\n"
            " - accept (closed: true), OR\n"
            " - make a counter-offer (next_offer) moving stepwise toward floor (prefer small concessions, e.g., 3-10% of the remaining gap), OR\n"
            " - ask a clarifying question (next_offer can be null in that case).\n\n"
            "Provide the JSON object now."
        )

        # Call the LLM. Use response_format to request a JSON object.
        # Note: your SDK may differ; adapt accordingly.
        response = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            # Ensure the model returns a JSON object only (SDK-specific param shown per your earlier usage)
            response_format={"type": "json_object"}
        )

        # Parse the LLM output; different SDKs place text differently. Adjust if needed.
        # Example: response.choices[0].message.content  (as in your earlier examples)
        content_text = response.choices[0].message.content
        try:
            llm_result = json.loads(content_text)
        except Exception as e:
            # If JSON parsing fails, stop negotiation and log for debugging
            logging.exception("Failed to parse LLM JSON response. Content was:\n%s", content_text)
            return {
                "error": "Failed to parse LLM response.",
                "llm_raw": content_text
            }

        # Validate required keys
        next_stage = llm_result.get("next_stage")
        memory_update = llm_result.get("memory_update", {})
        reply = llm_result.get("reply", "")
        next_offer = llm_result.get("next_offer", None)
        closed_flag = bool(llm_result.get("closed", False))

        # Append assistant reply to conversation_history
        # conversation_history.append({"user": user_reply_text, "assistant": reply})

        # Merge memory updates
        # memory.update(memory_update)

        # If LLM says closed or buyer accepted, we finish
        if closed_flag:
            update_stage(user_id, "post_sale_followup")
            final_offer = next_offer if isinstance(next_offer, (int, float)) else current_offer
            closed = True
            return {
                "listed_price": listed_price,
                "starting_offer": starting_offer,
                "final_offer": int(round(final_offer)),
                "closed": True,
                "conversation_history": conversation_history,
                "assistant_message": reply,
                "memory_update": memory_update,
            }

        # If LLM asked a question (next_offer is null), the assistant expects a buyer reply.
        if next_offer is None:
            # In a live system, you'd now wait for the buyer's reply and call this function again or continue the flow.
            # Here we return a partial state (assistant asked a clarifying question).
            return {
                "listed_price": listed_price,
                "starting_offer": starting_offer,
                "current_offer": current_offer,
                "closed": False,
                "conversation_history": conversation_history,
                "awaiting_user": True,
                "memory_update": memory_update,
                "assistant_message": reply,
            }

        # If LLM provided a numeric next_offer, normalize and enforce floor
        try:
            proposed_offer = int(round(float(next_offer)))
        except Exception:
            # Invalid offer format: stop and return
            return {
                "error": "LLM returned invalid next_offer",
                "llm_result": llm_result
            }

        # Enforce negotiation floor server-side: do not accept below floor
        if proposed_offer < negotiation_floor:
            proposed_offer = negotiation_floor

        # If proposed_offer is same as current or worse (higher than listed price when buyer won't accept),
        # we limit to a small concession step toward the floor
        if proposed_offer < current_offer:
            # valid concession (assistant is lowering seller's price)
            current_offer = proposed_offer
        else:
            # assistant didn't concede; lower by a small step toward floor automatically (5% of remaining gap)
            gap = current_offer - negotiation_floor
            if gap <= 0:
                # can't move further
                return {
                    "listed_price": listed_price,
                    "starting_offer": starting_offer,
                    "final_offer": current_offer,
                    "closed": False,
                    # "conversation_history": conversation_history,
                    "assistant_message": reply,
                    "memory_update": memory_update,
                    "note": "Reached negotiation floor or no further concessions."
                }
            concession = max(1, int(round(gap * 0.05)))  # 5% step, at least 1
            current_offer = current_offer - concession

        # Append the seller counter-offer to conversation_history as assistant's last action (if not already reflected)
        # conversation_history.append({"user": user_reply_text, "assistant": f"Counter-offer: KES {current_offer:,}"})

        # If we've reached or passed the halfway point, treat as 'closing attempt' in next iteration
        if current_offer <= negotiation_floor or current_offer <= (listed_price * 0.5):
            # Final closing attempt: ask buyer if they accept
            final_prompt = (
                "Final offer (closing attempt): "
                f"KES {current_offer:,}. Ask the buyer to confirm acceptance to close the sale. "
                "Return the JSON object with closed true if buyer accepts; otherwise closed false and next_offer null."
            )
            # Send a small prompting call (we could reuse the loop, but to keep demonstration short, return state)
            return {
                "listed_price": listed_price,
                "starting_offer": starting_offer,
                "current_offer": current_offer,
                "closed": False,
                # "conversation_history": conversation_history,
                "assistant_message": f"Counter-offer: KES {current_offer:,}",
                "memory_update": memory_update,
                "assistant_message": f"Our best and final offer is KES {current_offer:,}. If you’re ready, please visit us at {best_geolocation} to complete the purchase."

            }

    # If max rounds exhausted
    return {
        "listed_price": listed_price,
        "starting_offer": starting_offer,
        "final_offer": current_offer,
        "closed": closed,
        "conversation_history": conversation_history,
        "assistant_message": f"Our best and final offer is KES {current_offer:,}. If you’re ready, please visit us at {best_geolocation} to complete the purchase.",
        "memory_update": memory_update,
        "note": "Max negotiation rounds reached"
    }



def generate_answer_v1(user_id, message):
    logging.info(f"User ({user_id}) says: {message}")
    
    if user_id not in conversation_memory:
        conversation_memory[user_id]["stage"] = "awaiting_model"

    # memory = conversation_memory[user_id]
    memory = load_memory_from_db(user_id)
    logging.info(f"Current memory before processing: {memory}")

    # -----------------------
    # If awaiting budget
    # -----------------------
    if memory['stage'] == 'post_sale_followup':
        filtered_memory_post_follow_up = {k: v for k, v in memory.items() if k != 'stage' and k != 'budget'}
        df_post_follow_up = filter_cars_tool(user_id=user_id, **filtered_memory_post_follow_up)
        geolocation_col = normalize_colname(df_post_follow_up, "geolocation")
        best_geolocation = df_post_follow_up[geolocation_col].iloc[0] if not df_post_follow_up.empty else "our dealership"
        return f"Thank you for choosing us! Ready to finalize the deal? Please [Visit us anytime here]({best_geolocation})."



    if memory["stage"] == "awaiting_budget":
        # import pdb;pdb.set_trace()
        user_responses = [x['user'] for x in conversation_history if 'user' in x]
        budget = extract_budget(' '.join(user_responses + [message]))
        if budget is None:
            logging.warning("Failed to parse budget from message.")
            return "What is your budget? Please enter it like '2,100,000' or '2.1m'."
        memory["budget"] = budget
        update_budget(user_id, budget)
        filtered_memory = {k: v for k, v in memory.items() if k != 'stage' and k != 'budget'}
        df_ = filter_cars_tool(user_id=user_id, **filtered_memory)
        negotiation_result = negotiate_price_from_memory(memory, df_, message)
        return negotiation_result.get("assistant_message") if "assistant_message" in negotiation_result else negotiation_result.get('conversation_history', [])[-1].get('assistant', "Thank you for your interest.")



    # -----------------------
    # Otherwise, ask LLM to detect info and next stage
    # -----------------------
    prompt = f"""
    Conversation memory: {json.dumps(load_memory_from_db(user_id))}
    Customer message: "{message}"

    ### TASK
    1. Detect if the user is providing information for any of these fields: make, model, drive, body_type, color, budget
    2. Suggest the next stage of conversation: one of ["awaiting_model","awaiting_drive","awaiting_body_type","awaiting_colour","awaiting_budget","completed"]
    3. Update memory with any new info
    4. Generate a short, natural reply to continue narrowing
    5. Do not give examples or explanations.
    6. Ask the next user question based on the conversation memory given in order to reach "completed" stage.
    7. Ensure next question and generated reply do not clash or repeat.
    8. Do not give options or examples!!!!!!!

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
    # conversation_history.append({"user": message, "assistant": parsed.get("reply", "")})

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
