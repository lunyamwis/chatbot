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
from models import Base, engine, SessionLocal, ConversationMemory, ConversationHistory
# import pandasai as pai
# from pandasai_litellm.litellm import LiteLLM

# # Initialize LiteLLM with your OpenAI model
# pandas_llm = LiteLLM(model="gpt-4.0-mini", api_key=os.getenv("OPENAI_API_KEY")).strip()

# # Configure PandasAI to use this LLM
# pai.config.set({
#     "llm": pandas_llm
# })

import sqlite3
from contextlib import closing
import pandas as pd
import os
import gradio as gr
from huggingface_hub import login
from smolagents import CodeAgent
from smolagents import LiteLLMModel
# login(os.getenv("HF_TOKEN").strip())


# Load your car sales CSV



DB_PATH = "chatbot.db"


def load_conversation_history(user_id: str) -> list:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT * FROM conversation_history
            WHERE user_id='{user_id}';
        """)
        rows = cursor.fetchall()

        # Get column names
        columns = [desc[0] for desc in cursor.description]

        # Convert each row to a dictionary
        results = [dict(zip(columns, row)) for row in rows]

        return results

llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY").strip())
# import litellm
# litellm._turn_on_debug()
# -----------------------
# Logging setup
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------
# Load CSV
# -----------------------
df = pd.read_csv("data/botdata.csv")

# -----------------------
# Conversation memory
# -----------------------
user_id = str(uuid.uuid4())
print(f"Generated user_id: {user_id}")



def init_db():
    Base.metadata.create_all(bind=engine)

if os.path.exists(DB_PATH) == False:
    init_db()
    
# ---------- Safe integer conversion ----------
def to_int(value):
    try:
        if value is None:
            return None
        return int(value)
    except:
        return None



# ---------- Load memory from DB ----------
def load_memory_from_db(user_id: str) -> dict:
    session = SessionLocal()
    record = session.query(ConversationMemory).filter_by(user_id=user_id).first()
    session.close()

    if not record:
        return {
            "make": None,
            "model": None,
            "colour": None,
            "body_type": None,
            "year": None,
            "engine_cc": 0,
            "drive": None,
            "fuel": None,
            "mileage": 0,
            "transmission": None,
            "doors": 0,
            "price": 0,
            "selling_price": 0,
            "location": None,
            "geolocation": None,
            "budget": 0,
            "stage": "rapport_building",
            "next_stage": None,
            "phone_number": None,
            "email": None
        }

    return {
        "make": record.make,
        "model": record.model,
        "chassis_no": record.chassis_no,
        "colour": record.colour,
        "body_type": record.body_type,
        "year": record.year,
        "engine_cc": record.engine_cc,
        "drive": record.drive,
        "fuel": record.fuel,
        "mileage": record.mileage,
        "transmission": record.transmission,
        "doors": record.doors,
        "price": record.price,
        "selling_price": record.selling_price,
        "location": record.location,
        "geolocation": record.geolocation,
        "budget": record.budget,
        "stage": record.stage,
        "next_stage": record.next_stage,
        "phone_number": record.phone_number,
        "email": record.email
    }


# ---------- Save memory to DB ----------
def save_memory_to_db(user_id: str, memory: dict):
    session = SessionLocal()

    record = session.query(ConversationMemory).filter_by(user_id=user_id).first()
    if not record:
        record = ConversationMemory(user_id=user_id)

    # String fields — only overwrite if memory has a non-empty value
    string_fields = [
        "make", "model", "chassis_no", "colour", "body_type",
        "drive", "fuel", "transmission", "location", "geolocation",
        "stage", "next_stage", "phone_number", "email","features","negotiation_state","car_suggestions"
    ]

    for field in string_fields:
        value = memory.get(field)
        if value not in [None, ""]:  # only overwrite if value exists
            setattr(record, field, str(value))

    # Integer/number fields — only overwrite if memory has a non-zero value
    int_fields = [
        "year", "engine_cc", "mileage", "doors", "price",
        "selling_price", "budget"
    ]

    for field in int_fields:
        value = memory.get(field)
        if value not in [None, 0, ""]:
            setattr(record, field, to_int(value))

    session.add(record)
    session.commit()
    session.close()



def clear_memory_in_db():
    session = SessionLocal()
    try:
        deleted = (
            session.query(ConversationMemory)
            .filter_by(user_id=user_id)
            .delete()
        )
        session.commit()
        return f"✅ Cleared {deleted} memory records"
    except Exception as e:
        session.rollback()
        return f"❌ Error clearing memory: {str(e)}"
    finally:
        session.close()



def get_first_message(user_id: str):
    """
    Retrieve the first message for a given user from ConversationHistory.
    Returns None if no message exists.
    """
    session = SessionLocal()
    first_message = (
        session.query(ConversationHistory)
        .filter_by(user_id=user_id)
        .order_by(ConversationHistory.timestamp.asc())
        .first()
    )
    session.close()
    return first_message

def append_conversation_history(user_id: str, user_msg: str, assistant_msg: str):
    session = SessionLocal()

    entry = ConversationHistory(
        user_id=user_id,
        user_message=user_msg,
        assistant_message=assistant_msg
    )

    session.add(entry)
    session.commit()
    session.close()





def clear_conversation_history():
    session = SessionLocal()
    session.query(ConversationHistory).filter_by(user_id=user_id).delete()
    session.commit()
    session.close()


conversation_history = load_conversation_history(user_id)
# -----------------------
# Helpers
# -----------------------
def normalize_colname(df, col):
    for c in df.columns:
        if c.lower() == col.lower():
            return c
    return None






def choose_route_to_take_for_user_message(user_message: str, memory: dict):
    """
    Generate prompt for car negotiation engine. memory is a dictionary.
    NO tool calls - pure LLM reasoning with negotiation logic.
    """
    prompt = f"""
    choose which is the best route to go:
    return in json format only
    if {user_message} contains asking for price:
    {{
        "route": "price_enquiry"
    }}
    if {user_message} contains the word budget or speaking about budget or has anything to do with negotiation of price:
    {{
        "route": "price_negotiation"
    }}
    
    """
    
    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": prompt
            }
        ],
        response_format={"type": "json_object"}
    )
    print(f"Route choice response: {json.loads(response.choices[0].message.content)}")

    return json.loads(response.choices[0].message.content)




def price_enquiry_agent(user_message: str):
    prompt = f"""
    You are a car sales assistant. Answer the user's price enquiry based on the dataset of cars available.
    Use the dataset to find relevant cars and their prices.
    Provide a concise and informative response.
    User message: {user_message}
    Goal is to ask for the budget after providing price information.

    """
    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message}
        ],
        response_format={"type": "text"}
    )
    return response.choices[0].message.content








def call_openai_to_json(user_message=None):
    """
    Call the LLM to extract memory updates, next stage, and generate a reply.
    Integrates filtering at every stage using a tool.
    """
    analyze_conversation_history_and_update_context(conversation_history, user_id, user_message)
    
    # resp = choose_route_to_take_for_user_message(user_message, memory)
    # if resp.get("route") == "price_negotiation":
    #     reply = {"reply": negotiate_price_agent(memory=load_memory_from_db(user_id),user_message=user_message)}
    # elif resp.get("route") == "price_enquiry":
    #     reply = {"reply": price_enquiry_agent(user_message=user_message)}
    # elif resp.get("route") == "vehicle_enquiry":
    #     reply = vehicle_enquiry_agent(user_message=user_message)
    reply = vehicle_enquiry_agent(user_message=user_message)
    return reply
    
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



def update_memory_tool(memory: dict,) -> dict:
    """
    Update conversation memory without overwriting existing values with None.
    """
    save_memory_to_db(user_id, memory)
    print(f"Memory updated for {user_id}: {memory}")



def analyze_conversation_history_and_update_context(conversation_history: list, user_id: str, user_message: str):
    # import pdb;pdb.set_trace()
    """
    Analyze the conversation history to update context/memory.
    """
    prompt = f"""
        Analyze the following conversation between a user and a car sales assistant.

        Your task:
        1. Extract user preferences and relevant context.
        2. Only include fields that can be mapped to the JSON memory structure.
        3. Get the price of the car based on the conversation history based on what the assistant has mentioned.
        4. Get the features from the conversation history or from the {user_message}.
        5. If a field is not mentioned, output it as null.
        6. The **make** should be one of the following options: `{df[normalize_colname(df, "make")].dropna().unique().tolist()}`
        7. The **model** should be one of the following options: `{df.groupby('MAKE')['MODEL'].unique().apply(list).to_dict()}`
        8. The **location** should be chosen from: `{df.groupby('MODEL')['LOCATION'].unique().apply(list).to_dict()}`
        9. Using the conversation history, if a car has only one available color, automatically set the colour field. Similarly, if it comes with only one fuel type, set the fuel field accordingly
        11. Respond ONLY with a valid JSON object matching this schema:

        {{
            
            'make': null,
            'model': null,
            'chassis_no': null,
            'colour': null,
            'body_type': null,
            'year': null,
            'engine_cc': null,
            'drive': null,
            'fuel': null,
            'mileage': null,
            'transmission': null,
            'doors': null,
            'price': null,
            'selling_price': null,
            'location': null,
            'geolocation': null,
            'budget': null,
            'phone_number': null,
            'email': null,
            'features': null

        }}

        Conversation History:
        {json.dumps(load_conversation_history(user_id))}
        """


    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message}
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "update_memory_tool",
                "description": "Analyze the conversation and update the entire memory dictionary for the user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "make": {"type": "string"},
                        "model": {"type": "string"},
                        "chassis_no": {"type": "string"},
                        "colour": {"type": "string"},
                        "body_type": {"type": "string"},
                        "year": {"type": "integer"},
                        "engine_cc": {"type": "integer"},
                        "drive": {"type": "string"},
                        "fuel": {"type": "string"},
                        "mileage": {"type": "integer"},
                        "transmission": {"type": "string"},
                        "doors": {"type": "integer"},
                        "price": {"type": "number"},
                        "selling_price": {"type": "number"},
                        "location": {"type": "string"},
                        "geolocation": {"type": "string"},
                        "budget": {"type": "number"},
                        "phone_number": {"type": "string"},
                        "email": {"type": "string"}

                    },
                }
            }
        }],
        tool_choice = {
            "type": "function",
            "function": {"name": "update_memory_tool"}
        },

        response_format={"type": "json_object"}
    )

    # Extract the structured output
    #import pdb;pdb.set_trace()
    context_update = response.choices[0].message.content
    print(f"Context update for {user_id}: {context_update}")
    tool_results = response.choices[0].message.tool_calls
    if tool_results:
        for result in tool_results:
            if result.function.name == "update_memory_tool":
                # import pdb;pdb.set_trace()
                args_str = result.function.arguments

                # Check if arguments exist
                if not args_str:
                    print("No arguments provided.")
                else:
                    try:
                        args_dict = json.loads(args_str)  # safely parse JSON

                        if not args_dict:  # empty dictionary
                            print("No updates to memory from conversation history.")
                        else:
                            update_memory_tool(args_dict)  # call your update function
                    except json.JSONDecodeError:
                        print("Failed to parse arguments as JSON:", args_str)










from smolagents.models import OpenAIServerModel


def refer_from_the_knowledgebase_tool(message):
    task = f"""
    **You are a car-sales assistant.**
    You receive:

    * A DataFrame `df` containing:
    **MAKE, MODEL, GRADE, CHASSIS NO, COLOUR, IMAGE_URL, ENGINE CC, MILEAGE, YEAR/MONTH, LOCATION, PRICE, FEATURES**

    * A user request:
    `{message}` 

    ## **Your Task**
    You can infer the model from the following memory:{load_conversation_history(user_id)}

    Never return a car outside the df given.
    Use pandas to return the **best matching cars** from the DataFrame based on the user's request.
    You must always return a **non-empty DataFrame**.

    ## **1. Data Cleaning (ALWAYS run first)**

    ```python
    df['ENGINE CC'] = pd.to_numeric(df['ENGINE CC'], errors='coerce')
    df['MILEAGE'] = pd.to_numeric(df['MILEAGE'], errors='coerce')
    df = df.dropna(subset=['ENGINE CC'])

    ## **2. Flexible Matching Rules**
    Match user criteria **loosely**, using ranges when applicable:
    * **Engine size:** ± **500 cc** if the user has strictly specified a value do not assume just return all cars incase user has not mentioned the value
    and skip qualifying based on engine size, so if they say low, high, that is vague, just return all cars
    * **Year:** ± **2 years** from 2018
    * **Mileage:** allow ±1,000 km if the user has strictly specified a value do not assume just return all cars incase user has not mentioned the value
    and skip qualifying based on mileage, so if they say low, high, that is vague, just return all cars
    * **Features:** case-insensitive substring match
    * **Make/Model:** partial match allowed (e.g. “Land Cruiser”, “LandCruiser”, “Toyota LC”) If the user requests a specific model and there are no exact matches, return all cars from the same make as a fallback.
    Example: if the user requests a specific model like "Mazda CX-5" and no exact match exists, return all cars from the same make (e.g., all Mazdas) as alternative options.
    
    You may use `str.contains`, `.between()`, and similarity/substring logic.
    You do **not** need exact equality unless the user clearly demands it.
    **Numerical fields (engine CC, year, mileage, price):** 
    - Apply ± tolerance if the user specifies a value.
    - If not specified, prefer the lowest value, but only **among cars that match Make/Model**.

    ## **3. Fallback Logic (When strict filters return zero results)**

    If no exact/range matches are found on exact matches use or (|) on the queries instead of and (&):

    Return the top 5 closest matches, ranked by:

    - First of all, apply fuzzy matching to handle spelling variations (e.g., “Mercedes”, “Mercedez”, “Mercdes”) by using a case-insensitive regex pattern.
    For example:
    patterns = ["Mercedes", "Mercedez", "Mercdes"]
    # Create a regex pattern to match any of the variants (case-insensitive)
    regex_pattern = "|".join(patterns)
    # Use this pattern to search the DataFrame
    closest_matches = df[df['MAKE'].str.contains(regex_pattern, case=False, na=False)]

    - Partial Make/Model similarity (highest priority) 
    If the user requests a specific model and there are no exact matches, return all cars from the same make as a fallback.
    Example: if the user requests a specific model like "Mazda CX-5" and no exact match exists, return all cars from the same make (e.g., all Mazdas) as alternative options.
    
    When matching models, allow partial matches so that a more specific model (e.g., "Toyota Crown Athlete") can match a general model name (e.g., "Toyota Crown") if an exact match is not found.
    Example: User requests "Toyota Crown Athlete". If no exact match exists, return cars labeled "Toyota Crown" as a close match.

    - Number of matched features
    - Engine CC closeness
    - Year closeness
    - Mileage closeness

    Always include a **short explanation** like:
    `"No exact match found; showing closest alternatives within range."`

    ## **4. Output Rules**

    * Always output a **DataFrame** (never empty). 
    * Return columns dynamically based on the user’s query (e.g., color or mileage), while always including the stock ID and model, which remains consistent. 
    * Never raise errors.
    * Keep the explanation short.
    * Only add the price when the user's request contains wording about price, budget, or 'how much'.

   

    """



    # model = LiteLLMModel(
    #     # model_id="gpt-4",
    #     # provider="openai",
    #     # api_key=os.getenv("OPENAI_API_KEY"),  # Make sure your token is in env.strip()
    #     # max_tokens=4096,
    #     # temperature=0.1,
    #     # provider="huggingface",
    #     model="openai/gpt-4o-mini",  # or one of the others
    #     temperature=0.1,
    #     max_tokens=2048,
    #     # timeout=60,
    #     # max_retries=1
    #     # api_key=os.getenv("OPENAI_API_KEY"),
    #     # provider="openai"
    # )
    model = OpenAIServerModel(model_id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY").strip())
    agent = CodeAgent(
        model=model,
        tools=[],
        max_steps=4,
        additional_authorized_imports=["pandas", "numpy","difflib"],
        verbosity_level=2
    )
    result = agent.run(
        task,
        additional_args={"df": df}
    )
    return str(result)


# def return_matches_as_text(parsed, matches, user_message=None, is_off_tool=False) -> str:

def vehicle_enquiry_agent(user_message=None) -> str:
    memory = load_memory_from_db(user_id)
    

    # Retrieve the first message
    first_message = None
    first_msg = get_first_message(user_id=user_id)
    if first_msg:
        print("User's first message:", first_msg.user_message)
        first_message = first_msg.user_message
    else:
        # No message exists; treat the incoming message as the first
        print("This is the user's first message:", user_message)
        first_message = user_message
    


    # resp = choose_route_to_take_for_user_message(user_message, memory)
    # if resp.get("route") == "price_negotiation":
    #     negotiation_reply = {"reply": negotiate_price_agent(memory=load_memory_from_db(user_id),user_message=user_message)}
    # elif resp.get("route") == "price_enquiry":
    #     price_enquiry_reply = {"reply": price_enquiry_agent(user_message=user_message)}



    next_question_prompt = f"""
    You are assisting in an interactive car sales conversation.
    The goal is to guide the user step-by-step to find their ideal vehicle.:

    Rules:

    If {first_message} or {user_message} contains any wording related to vehicle features, specifications, trims, mechanical details, comfort/safety/tech features, engine descriptions, or performance details, then:
        - trigger the refer_from_the_knowledgebase_tool to perform a car search and check its availability.            

    If {user_message} is a very brief ambigous answer like yes or no or sure or ofcourse or no problem, then:
        - study this conversation history:{json.dumps(load_conversation_history(user_id))} in order to give it context and meaning and then rephrase it before passing it to the refer_from_the_knowledgebase_tool.
        
    If {user_message} is going off topic politely return them to the topic
    Context:
    - The memory:
    {load_memory_from_db(user_id)}
    - The user's latest message is:
    "{user_message}"

    Rules:
    - The output must be a pure JSON object in this exact format:

    {{
        "next_question": "string",
        "next_stage": "string",
        "negotiation_state": {{
            "listing_price": {memory.get('price', 0)},
            "current_offer": budget,,
            "target_price": listing_price*0.90,
            "step_type": "slow"|"quick",
        }}
    }}
    """
    # import pdb;pdb.set_trace()
    next_question_response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": next_question_prompt
            },
            {"role": "user", "content": user_message}
        ],

        response_format={"type": "json_object"},
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "refer_from_the_knowledgebase_tool",
                    "description": "Answers factual questions about vehicles such as color, mileage, engine, body type, and fuel type. Use only when the user asks for factual vehicle attributes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The user's question that requires vehicle knowledge base lookup"
                            }
                        },
                        "required": ["message"]
                    }
                }
            },
            {
                "type": "function",
                "function":{
                    "name": "negotiate_car_price",
                    "description": "Negotiate the car price based on the user's budget (current offer), listing price, and car features.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                        "listing_price": {
                            "type": "number",
                            "description": "The official listing price of the car."
                        },
                        "features_count": {
                            "type": "integer",
                            "description": "Number of additional features the car has; affects step size in negotiation."
                        },
                        "current_offer": {
                            "type": ["number", "null"],
                            "description": "The user's budget for the car; if None, start negotiation from the listing price."
                        }
                        },
                        "required": ["listing_price", "features_count"]
                    }
                    }

            }

        ],
        tool_choice = "auto"
    )

    # import pdb;pdb.set_trace()
    print("Next question response:", next_question_response.choices[0].message)

    message = next_question_response.choices[0].message
    if message.content:
        json_message = json.loads(message.content) 
        # import pdb;pdb.set_trace()
        if "negotiation_state" in json_message:
            negotiation_state = json_message["negotiation_state"]
            memory["negotiation_state"] = negotiation_state
            save_memory_to_db(user_id, memory)
            print(f"Negotiation state updated for {user_id}: {negotiation_state}")

        if "features" in json_message:
            features = json_message["features"]
            memory["features"] = features
            save_memory_to_db(user_id, memory)
            print(f"Features updated for {user_id}: {features}")

        
    # Check if the LLM decided to call a function/tool
    tool_results = message.tool_calls
    if tool_results:
        for result in tool_results:
            if result.function.name == "negotiate_car_price":
                args = json.loads(result.function.arguments)
                tool_result = negotiate_car_price(
                    listing_price=args.get("listing_price", 0),
                    features_count=len(memory.get("features", [])),
                    current_offer=args.get("current_offer", None)
                )
                print(f"Tool result for negotiation: {tool_result}")
                memory["negotiation_state"] = tool_result
                save_memory_to_db(user_id, memory)
                print(f"Negotiation state updated for {user_id}")

                return {"reply": f"Based on your budget, here's our counteroffer: {tool_result['counteroffer']}. Would you like to proceed?"}
            if result.function.name == "refer_from_the_knowledgebase_tool":
                args = json.loads(result.function.arguments)
                tool_result = refer_from_the_knowledgebase_tool(args.get("message", ""))
                # import pdb;pdb.set_trace()

                print(f"Tool result for knowledge base query: {tool_result}")
                code_blocks = re.findall(r"```python(.*?)```", tool_result, re.DOTALL | re.IGNORECASE)
                if not code_blocks:
                    return {"reply": "The best matches found: " + tool_result}
                last_code = code_blocks[-1].strip()

                # Execute the code with 'df' and 'pd' injected
                exec_env = {'df': df, 'filtered_df': df, 'pd': pd}  # Inject both df and pandas module
                exec(last_code,  exec_env)

                # Access the resulting DataFrame
                if 'result_df' in exec_env:
                    result_df = exec_env['result_df'].head(2)
                    print("Result DataFrame:")
                    print(result_df)
                    tool_result = result_df.to_json(index=False)
                # You can return the tool result directly or integrate into your conversation
                # For example, feed it back to the LLM to generate a natural reply:
                memory["car_suggestions"] = tool_result
                save_memory_to_db(user_id, memory)
                print(f"Car suggestions updated for {user_id}")

                return {"reply": "The best matches found: " + tool_result}
            # elif result.function.name == "refer_from_the_knowledgebase_complex_queries_tool":
            #     args = json.loads(result.function.arguments)
            #     tool_result = refer_from_the_knowledgebase_complex_queries_tool(args.get("message", ""))
            #     print(f"Tool result for complex knowledge base query: {tool_result}")
            #     return {"reply": tool_result}

    else:
        next_question = json.loads(message.content)
        
        print("Memory after stage update:", load_memory_from_db(user_id))
        print(f"Next question and stage: {next_question['next_question']} | {next_question['next_stage']}")


        # memory["stage"] = parsed["next_stage"]
        # import pdb;pdb.set_trace()
        if next_question["next_stage"] == "awaiting_budget" and memory.get("colour") and memory.get("body_type") and memory.get("drive") and (memory.get("model") or memory.get("make")):
            next_question["reply"] = "What is your budget for the vehicle?"
        else:
            next_question["reply"] = next_question.get("next_question", "")

        return next_question




def negotiate_car_price(listing_price, features_count, current_offer=None):
    """
    Simple negotiation engine for cars.
    
    Rules:
    - Multiple features (>5): 2.5% steps down to 90% of listing_price
    - Few features (<=5): 5% steps down to 90% of listing_price
    - 90% is the BEST (final) price
    """
    target_price = listing_price * 0.90  # Always 10% off max
    
    # Determine step size based on features
    if features_count > 5:
        step_size = 0.025  # 2.5% slow steps
    else:
        step_size = 0.05   # 5% quick steps
    
    # If no current offer, start from listing price
    if current_offer is None:
        current_price = listing_price
    else:
        current_price = current_offer
    
    # Calculate next counteroffer (step down)
    next_price = current_price * (1 - step_size)
    
    # Never go below 90%
    if next_price <= target_price:
        next_price = target_price
    
    # Response format
    step_description = "slow" if step_size == 0.025 else "quick"
    return {
        "current_price": round(current_price, 2),
        "counteroffer": round(next_price, 2),
        "target_price": round(target_price, 2),
        "step_type": step_description,
        "features_count": features_count,
        "done": next_price <= target_price,
        "deal_text": f"Counter: ${next_price} ({step_description} step)"
    }



def negotiate_price_agent(memory, user_message):
    """
    STRICT rule enforcement - no LLM creativity allowed.
    """
    prompt = f"""
    You are a car sales negotiation assistant. FOLLOW THESE RULES EXACTLY - NO DEVIATIONS.

    MANDATORY CALCULATION RULES (MATH REQUIRED):
    1. listing_price = {memory.get('price', 0)}
    2. features_count = len({memory.get('features', [])})
    3. target_price = listing_price * 0.90  # ALWAYS 90%
    4. step_size = 0.025 if features_count > 5 else 0.05  # 2.5% or 5%
    5. start_price = memory['negotiation_state']['current_offer'] or listing_price
    6. counteroffer = start_price * (1 - step_size)
    7. if counteroffer < target_price: counteroffer = target_price
    8. step_type = "slow" if step_size == 0.025 else "quick"

    Context:
    - Memory: {json.dumps(memory, indent=2)}
    - User: "{user_message}"

    MATH EXAMPLE (FOLLOW THIS):
    listing_price=95000, features_count=2 (>5? NO → step_size=0.05)
    start_price=95000 (no current_offer)
    counteroffer=95000*(1-0.05)=90250
    target_price=95000*0.90=85500
    90250 > 85500 → counteroffer=90250, step_type="quick", done=False

    Output ONLY valid JSON - NO EXCEPTIONS:
    {{
        "response": "Friendly response mentioning EXACT counteroffer and step_type",
        "next_stage": "negotiating"|"deal_closed",
        "negotiation_state": {{
            "listing_price": {memory.get('price', 0)},
            "features_count": {len(memory.get('features', []))},
            "current_offer": {memory['negotiation_state']['current_offer'] if memory.get('negotiation_state') else None},
            "counteroffer": YOUR_CALCULATED_NUMBER,
            "target_price": listing_price*0.90,
            "step_type": "slow"|"quick",
            "done": counteroffer <= target_price
        }}
    }}

    CALCULATE NOW using the exact math above. Do not guess or invent numbers.
    """
    
    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)




def extract_budget(text: str):
    """
    Extract budget amount from user text.
    Returns budget as float or None if not found.
    """
    # Simple regex to find currency amounts (e.g., "1000000", "1,000,000", "KES 1,000,000")
    budget_patterns = [
        r'KES\s*([\d,]+)',
        r'KSh\s*([\d,]+)',
        r'([\d,]+)\s*KES',
        r'([\d,]+)\s*KSh',
        r'([\d,]+)'  # fallback to just numbers
    ]

    for pattern in budget_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace(',', '')
            try:
                return float(amount_str)
            except ValueError:
                continue

    return None



def asking_phone_number(message):
    resp = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly and helpful car sales assistant. "
                    f"Here is the full conversation history: {load_conversation_history(user_id=user_id)}. "
                    "Your task is to determine whether the user has provided their phone number. "
                    "If the message contains a valid phone number, return JSON with {'phone_number_given': 'true'}. "
                    "If not, politely ask them for their phone number and return {'phone_number_given': 'false'}. "
                    "Prompt the user for their phone number using the following format:{'question': 'Please provide your phone number so we can contact you regarding your car inquiry.'}"
                    "Respond in JSON format ONLY. "
                )
            },
            {
                "role": "user",
                "content": message
            }
        ],
        response_format={"type": "json_object"}
    )

    returned_content = resp.choices[0].message.content

    data = json.loads(returned_content)
    print(f"Phone number analysis result: {data}")
    return data




def return_options():
    pass



def generate_answer_v1(user_id, message):
    logging.info(f"User ({user_id}) says: {message}")
    
    
    # memory = conversation_memory[user_id]
    memory = load_memory_from_db(user_id)
    logging.info(f"Current memory before processing: {memory}")

    # -----------------------
    # If awaiting budget
    # -----------------------

    parsed = call_openai_to_json(user_message=message)
    print(f"Parsed LLM response: {parsed}")
    # update_memory(memory, parsed.get("memory_update", {}))
    # memory["stage"] = parsed.get("next_stage", memory["stage"])
    logging.info(f"Updated memory: {memory}")
    # append_conversation_history({"user": message, "assistant": parsed.get("reply", "")})

    # -----------------------
    # Filter top 5 cars for the current stage
    # -----------------------
    
    return parsed.get("reply", "Could you please clarify?")



# 1. If needed, continue the conversation by asking the user about their preferences for **['MAKE', 'MODEL', 'COLOUR', 'FUEL']**, requesting only the attributes that are missing from the current conversation memory: `{load_memory_from_db(user_id)}`. However, if `{answer}` shows that the requested item is out of stock or unavailable, inform the user that you can suggest alternatives and **do not ask for any missing attributes** in that case.
        
# 1. Begin by summarizing the information in {answer}. 
#         Then, continue the conversation by asking the user about their preferences for **['MAKE', 'MODEL', 'COLOUR', 'FUEL']**, 
#         requesting only the attributes that are missing from the current conversation memory: {load_memory_from_db(user_id)}. 
#         When summarizing, include only the first (highest-priority) alternative from {answer}.

# 1. If needed, continue the conversation by asking the user about their preferences for **['MAKE', 'MODEL', 'COLOUR', 'FUEL']**, requesting only the attributes that are missing from the current conversation memory: `{load_memory_from_db(user_id)}`. Also give user suggestions based on `{answer}`.
        

def rephrase_answer_prompt(customer_question: str, answer: str, conversation_history: list = conversation_history) -> str:
    return f"""
        You are Karimi, a knowledgeable, polite, and human-like sales assistant. 
        Your goal is to respond naturally and professionally to customer inquiries.

        Customer question: {customer_question}
        Proposed answer: {answer}
        Conversation history: {load_conversation_history(user_id)}

        Your task:
        
        Here is a clean and polished rephrasing:

        Rules

        1. Make the proposed answer sound friendly but straight to the point, concise, and human-like.
        2. Keep the final response under **60 words**.
        3. If from the proposed answer there is no exact match during a car search, share the alternatives given in the proposed answer and outline you are willing to import.
        4. Only share images when requested by the customer, and use the links from the IMAGE_URL column to do so.


        
        
        Tone examples to guide your style:
        Example: How much ?
        Karimi: "The 2018 Toyota Prado in Pearl colour is priced at 2.1 million KES."

        Example: Can I get an offroad vehicle?
        Karimi: “We have several SUVs available. Do you prefer Toyota or Nissan?”

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
        append_conversation_history(**{"user_id": user_id, "user_msg": customer_question, "assistant_msg": "[FLAGGED] Robot accusation detected."})
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
    # append_conversation_history(**{"user_id": user_id, "user_msg": customer_question, "assistant_msg": answer})
    # return answer

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

    append_conversation_history(**{"user_id": user_id, "user_msg": customer_question, "assistant_msg": gpt_response})
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
