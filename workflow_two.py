from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import logging
import asyncio
import json
import sqlite3
from datetime import datetime
from dataclasses import dataclass, asdict
import uuid
import os
import re
import uuid

logger = logging.getLogger(__name__)

# === DATA MODELS ===
@dataclass
class ConversationTurn:
    id: str
    session_id: str
    user_message: str
    bot_response: str
    timestamp: datetime
    event_type: str
    metadata: Dict[str, Any]
    sentiment: str = "neutral"
    confidence: float = 0.0

@dataclass
class SessionSummary:
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    vehicle_model: Optional[str]
    customer_intent: str
    negotiation_rounds: int
    final_status: str
    handoff_reason: Optional[str]
    summary_text: str

# === VECTOR STORE KNOWLEDGE BASE ===
class CarKnowledgeBase:
    def __init__(self, csv_path: str, persist_dir: str = "./chroma_database"):
        self.csv_path = csv_path
        self.persist_dir = persist_dir
        self.embed_model = OpenAIEmbedding()
        self.llm = OpenAI(model="gpt-3.5-turbo")
        self.vector_index = None
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize vector store from CSV data"""
        try:
            # Load CSV data
            df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded CSV with {len(df)} vehicles")
            
            # Create ChromaDB client
            chroma_client = chromadb.PersistentClient(path=self.persist_dir)
            chroma_collection = chroma_client.get_or_create_collection("car_inventory")
            
            # Create vector store
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create documents from CSV data
            documents = []
            for _, row in df.iterrows():
                doc_text = self._create_document_text(row)
                document = Document(
                    text=doc_text,
                    metadata={
                        "model": row['model'],
                        "price": float(row['price']),
                        "category": row.get('category', 'sedan'),
                        "profit_margin": float(row['profit_margin']),
                        "inventory_status": row.get('inventory_status', 'in_stock'),
                        "features": row.get('features', ''),
                        "year": int(row.get('year', 2024))
                    }
                )
                documents.append(document)
            
            # Create index
            self.vector_index = VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context,
                embed_model=self.embed_model
            )
            
            logger.info("Knowledge base initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {e}")
            raise

    def _create_document_text(self, row: pd.Series) -> str:
        """Create a detailed document text for semantic search, parsing the JSON in 'features'."""
        features = json.loads(row['features'].replace("'", '"'))  # Parse JSON string

        return f"""
        Vehicle Model: {row['model']}
        Year: {row['year']}
        Price: KES{float(row['price']):,}
        Category: {row.get('category', 'sedan')}
        Make: {features.get('MAKE', 'N/A')}
        Colour: {features.get('COLOUR', 'N/A')}
        Mileage: {features.get('MILEAGE', 'N/A')} km
        Fuel: {features.get('FUEL', 'N/A')}
        Transmission: {features.get('TRANSMISSION', 'N/A')}
        Engine CC: {features.get('ENGINE CC', 'N/A')}
        Drive: {features.get('DRIVE', 'N/A')}
        Location: {features.get('LOCATION', 'N/A')}
        Chassis No: {features.get('CHASSIS NO', 'N/A')}
        Doors: {features.get('DOORS', 'N/A')}
        Profit Margin: {float(row['profit_margin']) * 100:.1f}%

        Description: This {row.get('year')} {row['model']} is available for purchase.
        It is a {row.get('category')} with {features.get('MILEAGE', 'N/A')} km,
        powered by {features.get('FUEL', 'N/A')} and features {features.get('TRANSMISSION', 'N/A')} transmission.
        The vehicle is located in {features.get('LOCATION', 'N/A')} and comes in {features.get('COLOUR', 'N/A')} colour.
        """


    def query_vehicles(self, query: str, filters: Optional[Dict] = None, k: int = 5) -> List[Dict]:
        if not self.vector_index:
            raise ValueError("Knowledge base not initialized")

        try:
            query_engine = self.vector_index.as_query_engine(
                similarity_top_k=k,
                llm=self.llm
            )

            enhanced_query = self._enhance_query_with_filters(query, filters)
            response = query_engine.query(enhanced_query)

            results = []
            for node in response.source_nodes:
                metadata = node.metadata
                features = json.loads(metadata['features'].replace("'", '"'))  # Parse JSON string

                vehicle_data = {
                    'model': metadata.get('model', ''),
                    'price': metadata.get('price', 0),
                    'category': metadata.get('category', ''),
                    'year': metadata.get('year', 2024),
                    'profit_margin': metadata.get('profit_margin', 0),
                    'inventory_status': metadata.get('inventory_status', ''),
                    'features': features,  # Parsed JSON as dict
                    'similarity_score': node.score,
                    'text_snippet': node.node.get_content()[:200] + "..."
                }
                results.append(vehicle_data)

            return sorted(results, key=lambda x: x['similarity_score'], reverse=True)

        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            return []

    
    def _enhance_query_with_filters(self, query: str, filters: Optional[Dict]) -> str:
        """Enhance query with filter context"""
        if not filters:
            return query
        
        filter_parts = []
        if filters.get('max_price'):
            filter_parts.append(f"under KES{filters['max_price']:,}")
        if filters.get('category'):
            filter_parts.append(f"{filters['category']} type")
        if filters.get('min_year'):
            filter_parts.append(f"year {filters['min_year']} or newer")
        
        if filter_parts:
            return f"{query} {' '.join(filter_parts)}"
        return query
    
    def get_vehicle_by_model(self, model: str) -> Optional[Dict]:
        """Get specific vehicle information by model name"""
        try:
            results = self.query_vehicles(f"exact model {model}", k=1)
            if results and results[0]['model'].lower() == model.lower():
                return results[0]
            return None
        except Exception as e:
            logger.error(f"Error getting vehicle by model: {e}")
            return None
    
    def get_similar_vehicles(self, model: str, k: int = 3) -> List[Dict]:
        """Get similar vehicles based on model"""
        try:
            vehicle = self.get_vehicle_by_model(model)
            if not vehicle:
                return []
            
            # Query for similar vehicles in same category/price range
            query = f"{vehicle['category']} vehicles similar to {model} around KES{vehicle['price']:,}"
            return self.query_vehicles(query, k=k)
        except Exception as e:
            logger.error(f"Error getting similar vehicles: {e}")
            return []
    
    
    def is_relevant_to_cars(self, query: str) -> bool:
        """Check relevance using LLM classification for better accuracy"""
        try:
            query_lower = query.lower().strip()
            
            # Quick keyword pre-filter for obvious cases
            obvious_car_terms = [
                "buy car", "purchase vehicle", "test drive", "car price",
                "vehicle features", "car inventory", "lease car", "finance vehicle"
            ]
            
            if any(term in query_lower for term in obvious_car_terms):
                return True
                
            obvious_off_topic = [
                "weather", "sports", "politics", "music", "recipe", "movie",
                "restaurant", "hotel", "medical", "stock market", "crypto"
            ]
            
            if any(term in query_lower for term in obvious_off_topic):
                return False
            
            # Use LLM for classification
            classification_prompt = f"""
            Analyze if the following user query is related to car sales, vehicle purchasing, 
            or automotive topics. Respond with ONLY "CAR_RELATED" or "NOT_CAR_RELATED".
            
            Consider these as CAR_RELATED:
            - Buying, purchasing, or leasing vehicles
            - Vehicle prices, features, specifications
            - Test drives, inventory, availability
            - Financing, loans, payments for vehicles
            - Vehicle comparisons, reviews
            - Car maintenance, service (if related to purchase decision)
            
            Consider these as NOT_CAR_RELATED:
            - General automotive repair questions
            - Insurance claims (unless related to purchase)
            - Off-topic subjects (weather, sports, etc.)
            - Other product categories (houses, electronics, etc.)
            
            Query: "{query}"
            
            Classification:
            """
            
            # Use the LLM to classify
            llm = OpenAI(model="gpt-3.5-turbo")
            response = llm.complete(classification_prompt)
            classification = response.text.strip().upper()
            
            logger.info(f"Query classification: '{query}' -> {classification}")

            return True if "CAR_RELATED" == classification else False

        except Exception as e:
            logger.error(f"Error in LLM relevance classification: {e}")
            # Fallback to vector search with higher threshold
            try:
                results = self.query_vehicles(query, k=1)
                if results and results[0]['similarity_score'] > 0.7:
                    return True
            except:
                pass
                
            # Final fallback: strict keyword matching
            strict_car_terms = [
                "buy", "purchase", "lease", "test drive", "car", "vehicle", "suv", 
                "truck", "sedan", "price", "cost", "finance", "loan", "payment"
            ]
            query_lower = query.lower()
            return any(term in query_lower for term in strict_car_terms)
        

    
# === CONVERSATION STORAGE ===
class ConversationStorage:
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                event_type TEXT NOT NULL,
                sentiment TEXT,
                confidence REAL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_summaries (
                session_id TEXT PRIMARY KEY,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                vehicle_model TEXT,
                customer_intent TEXT,
                negotiation_rounds INTEGER DEFAULT 0,
                final_status TEXT,
                handoff_reason TEXT,
                summary_text TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON conversation_turns(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversation_turns(timestamp)')
        
        conn.commit()
        conn.close()
    
    def store_conversation_turn(self, turn: ConversationTurn):
        """Store conversation turn"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversation_turns 
            (id, session_id, user_message, bot_response, timestamp, event_type, sentiment, confidence, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            turn.id,
            turn.session_id,
            turn.user_message,
            turn.bot_response,
            turn.timestamp.isoformat(),
            turn.event_type,
            turn.sentiment,
            turn.confidence,
            json.dumps(turn.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def store_session_summary(self, summary: SessionSummary):
        """Store session summary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO session_summaries 
            (session_id, start_time, end_time, vehicle_model, customer_intent, negotiation_rounds, final_status, handoff_reason, summary_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            summary.session_id,
            summary.start_time.isoformat(),
            summary.end_time.isoformat() if summary.end_time else None,
            summary.vehicle_model,
            summary.customer_intent,
            summary.negotiation_rounds,
            summary.final_status,
            summary.handoff_reason,
            summary.summary_text
        ))
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, session_id: str, limit: int = 50) -> List[ConversationTurn]:
        """Get conversation history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, session_id, user_message, bot_response, timestamp, event_type, sentiment, confidence, metadata
            FROM conversation_turns 
            WHERE session_id = ? 
            ORDER BY timestamp 
            LIMIT ?
        ''', (session_id, limit))
        
        turns = []
        for row in cursor.fetchall():
            turns.append(ConversationTurn(
                id=row[0],
                session_id=row[1],
                user_message=row[2],
                bot_response=row[3],
                timestamp=datetime.fromisoformat(row[4]),
                event_type=row[5],
                sentiment=row[6],
                confidence=row[7],
                metadata=json.loads(row[8]) if row[8] else {}
            ))
        
        conn.close()
        return turns

# === EVENT CLASSES ===
class UserMessageEvent(Event):
    user_message: str
    session_id: str

class KnowledgeQueryEvent(Event):
    query: str
    context: Dict[str, Any]
    response_text: str = ""

class PricingEvent(Event):
    vehicle_model: str
    customer_offer: float
    base_price: float
    negotiation_round: int = 0
    response_text: str = ""

class HumanHandoffEvent(Event):
    reason: str
    context: Dict[str, Any]
    urgency: str = "medium"
    response_text: str = ""

class ObjectionEvent(Event):
    objection_type: str
    customer_message: str
    context: Dict[str, Any]
    response_text: str = ""

class QualificationEvent(Event):
    customer_profile: Dict[str, Any]
    missing_info: List[str]
    response_text: str = ""

class ClosingEvent(Event):
    vehicle_model: str
    final_price: float
    customer_ready: bool
    response_text: str = ""

class LoopControlEvent(Event):
    loop_type: str
    iteration: int
    max_iterations: int
    context: Dict[str, Any]
    response_text: str = ""


# === STATE MANAGEMENT ===
class CarSalesState:
    def __init__(self, session_id: str, storage: ConversationStorage, knowledge_base: CarKnowledgeBase):
        self.session_id = session_id
        self.storage = storage
        self.knowledge_base = knowledge_base
        self.conversation_history: List[ConversationTurn] = []
        self.current_vehicle: Optional[str] = None
        self.customer_profile: Dict[str, Any] = {}
        self.negotiation_history: List[Dict] = []
        self.profit_guardrail: float = 0.10
        self.start_time: datetime = datetime.now()
    
    def add_conversation_turn(self, user_message: str, bot_response: str, event_type: str, 
                            metadata: Dict[str, Any] = None):
        """Add conversation turn"""
        turn = ConversationTurn(
            id=str(uuid.uuid4()),
            session_id=self.session_id,
            user_message=user_message,
            bot_response=bot_response,
            timestamp=datetime.now(),
            event_type=event_type,
            metadata=metadata or {}
        )
        
        self.conversation_history.append(turn)
        self.storage.store_conversation_turn(turn)




# === COMPLETE WORKFLOW ===

class CarSalesWorkflow(Workflow):
    def __init__(self, storage, knowledge_base):
        super().__init__()
        self.storage = storage
        self.knowledge_base = knowledge_base
        self.state = None
        self.llm = OpenAI(model="gpt-3.5-turbo")

    def _serialize_conversation_history(self, history: List[Any]) -> List[Dict]:
        """Convert conversation history to a JSON-serializable list of dicts."""
        serialized = []
        for turn in history:
            if hasattr(turn, 'to_dict'):  # If the object has a to_dict method
                serialized.append(turn.to_dict())
            elif hasattr(turn, '__dict__'):  # Fallback: Convert to dict
                serialized.append({k: str(v) for k, v in turn.__dict__.items() if not k.startswith('_')})
            else:
                serialized.append(str(turn))
        return serialized

    def _generate_llm_response(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Generate a response using LLM with JSON-serializable context."""
        # Serialize context to avoid TypeError
        serialized_context = {}
        if context:
            for key, value in context.items():
                if isinstance(value, list):
                    serialized_context[key] = self._serialize_conversation_history(value)
                elif hasattr(value, 'to_dict'):
                    serialized_context[key] = value.to_dict()
                elif hasattr(value, '__dict__'):
                    serialized_context[key] = {k: str(v) for k, v in value.__dict__.items() if not k.startswith('_')}
                else:
                    serialized_context[key] = str(value)

        full_prompt = f"""
        {prompt}
        {f"Context: {json.dumps(serialized_context, indent=2)}" if serialized_context else ""}
        """
        response = self.llm.complete(full_prompt)
        return response.text.strip()

    @step
    def start(self, event: StartEvent) -> UserMessageEvent:
        """Start the conversation with an LLM-generated greeting."""
        session_id = str(uuid.uuid4())
        self.state = CarSalesState(session_id, self.storage, self.knowledge_base)

        prompt = """
        Generate a friendly and engaging greeting for a car sales assistant.
        The greeting should be warm, inviting, and encourage the user to ask about vehicles.
        Keep it concise (1-2 sentences).
        """
        greeting = self._generate_llm_response(prompt)
        return UserMessageEvent(user_message=greeting, session_id=session_id)

    @step
    def handle_user_message(self, event: UserMessageEvent) -> Event:
        """Handle user messages dynamically using LLM."""
        user_message = event.user_message
        self.state.add_conversation_turn(
            user_message=user_message,
            bot_response="",
            event_type="user_message"
        )

        # --- Check for feature-specific questions ---
        feature = self._detect_feature_question(user_message)
        if feature and self.state.current_vehicle:
            vehicle = self.knowledge_base.get_vehicle_by_model(self.state.current_vehicle)
            if not vehicle:
                return self._respond_vehicle_not_found(event, self.state.current_vehicle)
            return self._respond_feature_question(event, vehicle, feature, user_message)

        # --- Extract intent and entities ---
        intent, entities = self._extract_intent_and_entities(user_message)

        # --- Generate LLM response based on intent ---
        if intent == "general_inquiry":
            return self._handle_general_inquiry(event, entities, user_message)
        elif intent == "specific_vehicle":
            return self._handle_specific_vehicle(event, entities, user_message)
        elif intent == "comparison":
            return self._handle_comparison(event, entities, user_message)
        elif intent == "follow_up":
            return self._handle_follow_up(event, entities, user_message)
        else:
            return self._handle_unclear_intent(event, user_message, intent, entities)

    def _detect_feature_question(self, user_message: str) -> Optional[str]:
        """Detect if the user is asking about a specific feature."""
        feature_keywords = {
            "mileage": ["mileage", "kilometers", "km", "how many miles"],
            "colour": ["colour", "color", "what colour", "what color", "available colours"],
            "fuel": ["fuel", "petrol", "diesel", "hybrid", "electric", "fuel type"],
            "transmission": ["transmission", "automatic", "manual", "gear"],
            "engine": ["engine", "cc", "engine size", "engine capacity"],
            "drive": ["drive", "4wd", "2wd", "awd", "four-wheel drive"],
            "location": ["location", "where is it", "available in", "city"],
            "make": ["make", "manufacturer", "brand"],
            "chassis": ["chassis", "chassis number"],
            "doors": ["doors", "how many doors"]
        }

        user_message_lower = user_message.lower()
        for feature, keywords in feature_keywords.items():
            if any(keyword in user_message_lower for keyword in keywords):
                return feature
        return None

    def _respond_feature_question(self, event: UserMessageEvent, vehicle: Dict, feature: str, user_message: str) -> Event:
        """Respond to feature-specific questions using LLM."""
        context = {
            "vehicle": vehicle,
            "feature": feature,
            "user_message": user_message,
            "conversation_history": self.state.conversation_history[-3:]  # Last 3 turns
        }
        prompt = f"""
        You are a helpful car sales assistant. Respond to the user's question about the {feature} of the {vehicle['model']}.
        Use the following vehicle details to craft your response:
        Model: {vehicle['model']}
        Year: {vehicle['year']}
        Price: KES{vehicle['price']:,}
        Features: {json.dumps(vehicle['features'], indent=2)}

        Guidelines:
        1. Be friendly, natural, and conversational.
        2. Use the vehicle details to provide accurate information.
        3. Encourage further questions or next steps (e.g., test drive, pricing discussion).
        4. Keep responses concise (1-2 sentences).

        User's question: "{user_message}"
        Your response:
        """
        response = self._generate_llm_response(prompt, context)
        return UserMessageEvent(user_message=response, session_id=event.session_id)

    def _extract_intent_and_entities(self, user_message: str) -> Tuple[str, Dict[str, Any]]:
        """Use LLM to extract intent and entities from user message."""
        prompt = f"""
        Analyze the following user message and extract:
        1. Intent (general_inquiry, specific_vehicle, comparison, follow_up, feature_question)
        2. Entities (e.g., model, category, models for comparison, feature)

        User Message: "{user_message}"

        Respond in JSON format:
        {{
            "intent": "...",
            "entities": {{...}}
        }}
        """
        response = self.llm.complete(prompt)
        try:
            result = json.loads(response.text.strip())
            return result["intent"], result["entities"]
        except Exception as e:
            logger.error(f"Error parsing LLM intent: {e}")
            return "unclear", {}

    def _handle_general_inquiry(self, event: UserMessageEvent, entities: Dict[str, Any], user_message: str) -> Event:
        """Handle general inquiries using LLM."""
        category = entities.get("category", "")
        vehicles = self.knowledge_base.query_vehicles(
            query=user_message,
            filters={"category": category},
            k=3
        )

        if not vehicles:
            return self._respond_no_vehicles_found(event, category, user_message)

        context = {
            "vehicles": vehicles,
            "category": category,
            "user_message": user_message,
            "conversation_history": self.state.conversation_history[-3:]
        }
        prompt = f"""
        You are a helpful car sales assistant. The user is asking about {category.lower()}s.
        Here are the available vehicles:
        {json.dumps([{
            "model": v['model'],
            "year": v['year'],
            "price": v['price'],
            "features": v['features']
        } for v in vehicles], indent=2)}

        Guidelines:
        1. Be friendly and engaging.
        2. Highlight key details (price, features, mileage, etc.).
        3. Encourage the user to ask follow-up questions or express interest.
        4. Keep responses concise (2-3 sentences).

        User's message: "{user_message}"
        Your response:
        """
        response = self._generate_llm_response(prompt, context)
        return UserMessageEvent(user_message=response, session_id=event.session_id)

    def _handle_specific_vehicle(self, event: UserMessageEvent, entities: Dict[str, Any], user_message: str) -> Event:
        """Handle inquiries about a specific vehicle using LLM."""
        model = entities.get("model", "")
        vehicle = self.knowledge_base.get_vehicle_by_model(model)
        if not vehicle:
            return self._respond_vehicle_not_found(event, model)

        self.state.current_vehicle = vehicle['model']
        context = {
            "vehicle": vehicle,
            "user_message": user_message,
            "conversation_history": self.state.conversation_history[-3:]
        }
        prompt = f"""
        You are a helpful car sales assistant. The user is asking about the {vehicle['model']} ({vehicle['year']}).
        Here are the vehicle details:
        {json.dumps(vehicle, indent=2)}

        Guidelines:
        1. Be enthusiastic and informative.
        2. Highlight key selling points (price, features, mileage, etc.).
        3. Encourage next steps (test drive, financing, comparisons).
        4. Keep responses concise (2-3 sentences).

        User's message: "{user_message}"
        Your response:
        """
        response = self._generate_llm_response(prompt, context)
        return UserMessageEvent(user_message=response, session_id=event.session_id)

    def _handle_comparison(self, event: UserMessageEvent, entities: Dict[str, Any], user_message: str) -> Event:
        """Handle comparison requests using LLM."""
        models = entities.get("models", [])
        vehicles = [
            self.knowledge_base.get_vehicle_by_model(model)
            for model in models
        ]

        if not all(vehicles):
            missing = [m for m in models if not self.knowledge_base.get_vehicle_by_model(m)]
            return self._respond_comparison_error(event, missing)

        context = {
            "vehicles": vehicles,
            "models": models,
            "user_message": user_message,
            "conversation_history": self.state.conversation_history[-3:]
        }
        prompt = f"""
        You are a helpful car sales assistant. The user wants to compare the {models[0]} and {models[1]}.
        Here are the details for each vehicle:
        {json.dumps([{
            "model": v['model'],
            "year": v['year'],
            "price": v['price'],
            "features": v['features']
        } for v in vehicles], indent=2)}

        Guidelines:
        1. Provide a clear, concise comparison.
        2. Highlight key differences (price, features, mileage, etc.).
        3. Encourage the user to ask follow-up questions or express preference.
        4. Keep responses friendly and engaging (2-3 sentences).

        User's message: "{user_message}"
        Your response:
        """
        response = self._generate_llm_response(prompt, context)
        return UserMessageEvent(user_message=response, session_id=event.session_id)

    def _handle_follow_up(self, event: UserMessageEvent, entities: Dict[str, Any], user_message: str) -> Event:
        """Handle follow-up questions using LLM."""
        model = entities.get("model", "")
        vehicle = self.knowledge_base.get_vehicle_by_model(model)
        if not vehicle:
            return self._respond_vehicle_not_found(event, model)

        self.state.current_vehicle = vehicle['model']
        context = {
            "vehicle": vehicle,
            "previous_vehicle": self.state.current_vehicle,
            "user_message": user_message,
            "conversation_history": self.state.conversation_history[-3:]
        }
        prompt = f"""
        You are a helpful car sales assistant. The user is asking about the {vehicle['model']} ({vehicle['year']})
        as a follow-up to discussing the {self.state.current_vehicle}.
        Here are the details for the {vehicle['model']}:
        {json.dumps(vehicle, indent=2)}

        Guidelines:
        1. Acknowledge the previous vehicle discussed.
        2. Highlight key details of the new vehicle.
        3. Encourage comparison or next steps (test drive, pricing, etc.).
        4. Keep responses concise and engaging (2-3 sentences).

        User's message: "{user_message}"
        Your response:
        """
        response = self._generate_llm_response(prompt, context)
        return UserMessageEvent(user_message=response, session_id=event.session_id)

    def _handle_unclear_intent(self, event: UserMessageEvent, user_message: str, intent: str, entities: Dict[str, Any]) -> Event:
        """Handle unclear intents using LLM."""
        context = {
            "user_message": user_message,
            "intent": intent,
            "entities": entities,
            "conversation_history": self.state.conversation_history[-3:]
        }
        prompt = f"""
        You are a helpful car sales assistant. The user's intent is unclear.
        User's message: "{user_message}"
        Extracted intent: {intent}
        Extracted entities: {json.dumps(entities)}

        Guidelines:
        1. Politely ask for clarification.
        2. Suggest possible next steps (e.g., browsing inventory, asking about specific models).
        3. Keep responses friendly and helpful (1-2 sentences).

        Your response:
        """
        response = self._generate_llm_response(prompt, context)
        return UserMessageEvent(user_message=response, session_id=event.session_id)

    def _respond_no_vehicles_found(self, event: UserMessageEvent, category: str, user_message: str) -> Event:
        """Respond when no vehicles match the query using LLM."""
        context = {
            "category": category,
            "user_message": user_message,
            "conversation_history": self.state.conversation_history[-3:]
        }
        prompt = f"""
        You are a helpful car sales assistant. No vehicles were found matching the user's request for {category.lower()}s.
        User's message: "{user_message}"

        Guidelines:
        1. Apologize and explain that no matches were found.
        2. Offer to adjust search criteria or connect with a specialist.
        3. Keep responses empathetic and helpful (1-2 sentences).

        Your response:
        """
        response = self._generate_llm_response(prompt, context)
        return UserMessageEvent(user_message=response, session_id=event.session_id)

    def _respond_vehicle_not_found(self, event: UserMessageEvent, model: str) -> Event:
        """Respond when a specific vehicle is not found using LLM."""
        context = {
            "model": model,
            "conversation_history": self.state.conversation_history[-3:]
        }
        prompt = f"""
        You are a helpful car sales assistant. The {model} was not found in the inventory.
        Guidelines:
        1. Apologize and explain that the vehicle is not available.
        2. Offer to check other models or connect with a specialist.
        3. Keep responses empathetic and helpful (1-2 sentences).

        Your response:
        """
        response = self._generate_llm_response(prompt, context)
        return UserMessageEvent(user_message=response, session_id=event.session_id)

    def _respond_comparison_error(self, event: UserMessageEvent, missing_models: List[str]) -> Event:
        """Respond when one or more vehicles in a comparison are not found using LLM."""
        context = {
            "missing_models": missing_models,
            "conversation_history": self.state.conversation_history[-3:]
        }
        prompt = f"""
        You are a helpful car sales assistant. The following models were not found: {', '.join(missing_models)}.
        Guidelines:
        1. Apologize and explain that the models are not available.
        2. Offer to check other models or connect with a specialist.
        3. Keep responses empathetic and helpful (1-2 sentences).

        Your response:
        """
        response = self._generate_llm_response(prompt, context)
        return UserMessageEvent(user_message=response, session_id=event.session_id)

    @step
    def escalate_to_human(self, reason: str, context: Dict[str, Any], previous_event: Event) -> StopEvent:
        """Hand off to a human agent with an LLM-generated message."""
        serialized_context = {}
        for key, value in context.items():
            if isinstance(value, list):
                serialized_context[key] = self._serialize_conversation_history(value)
            else:
                serialized_context[key] = str(value)

        prompt = f"""
        You are a helpful car sales assistant. You need to escalate the conversation to a human specialist.
        Reason: {reason}
        Context: {json.dumps(serialized_context, indent=2)}

        Guidelines:
        1. Politely explain that you’re connecting the user to a specialist.
        2. Mention the reason for the handoff (e.g., complex question, unavailable vehicle).
        3. Keep responses friendly and reassuring (1-2 sentences).

        Your response:
        """
        response = self._generate_llm_response(prompt, serialized_context)
        self.state.add_conversation_turn(
            user_message="",
            bot_response=response,
            event_type="human_handoff"
        )
        return StopEvent(
            result={
                "status": "handoff",
                "reason": reason,
                "context": context,
                "message": response
            }
        )



# === SAMPLE CSV STRUCTURE ===
"""
Create a CSV file named 'car_inventory.csv' with these columns:
model,price,category,profit_margin,features,year,inventory_status

Example rows:
camry,27500,sedan,0.15,"leather seats, sunroof, navigation",2024,in_stock
civic,24000,sedan,0.12,"backup camera, apple carplay, lane assist",2024,in_stock
cr-v,32500,suv,0.18,"awd, heated seats, premium audio",2024,low_stock
rav4,29500,suv,0.16,"safety sense, spacious interior",2024,in_stock
f-150,45000,truck,0.20,"towing package, 4x4, crew cab",2024,in_stock
"""

# === USAGE EXAMPLE ===
async def main():
    """Demo the complete workflow"""
    
    # Initialize components
    knowledge_base = CarKnowledgeBase("data/motorvehicle_data.csv")
    storage = ConversationStorage()
    
    # Test messages
    test_messages = [
        "I'm interested in a Toyota Camry",
        "What's the price of a Honda Civic?",
        "I want to buy a CR-V for KES28,000",
        "Tell me about electric cars"  # This should trigger irrelevant query handling
    ]
    
    print("=== COMPLETE CAR SALES WORKFLOW DEMO ===\n")
    
    for i, message in enumerate(test_messages):
        print(f"Test {i+1}: {message}")
        
        workflow = CarSalesWorkflow(
            session_id=f"session_{i}",
            knowledge_base=knowledge_base,
            storage=storage
        )
        
        result = await workflow.run(StartEvent())
        print(f"Result: {result.result}\n")
        
        # Show conversation history
        history = storage.get_conversation_history(f"session_{i}")
        print(f"Conversation History ({len(history)} turns):")
        for turn in history:
            print(f"  {turn.event_type}: {turn.user_message} -> {turn.bot_response}")
        print()

if __name__ == "__main__":
    # Set your OpenAI API key
    import os
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"
    
    asyncio.run(main())