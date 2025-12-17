from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, DateTime, func


# Change DB URL here when switching databases
# SQLite testing:
DATABASE_URL = "sqlite:///chatbot.db"


engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()

class Comments(Base):
    __tablename__ = "comments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    comment_text = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class ConversationMemory(Base):
    __tablename__ = "conversation_memory"

    user_id = Column(String, primary_key=True)

    make = Column(String)
    model = Column(String)
    chassis_no = Column(String)
    colour = Column(String)
    body_type = Column(String)
    year = Column(Integer)
    engine_cc = Column(Integer)
    drive = Column(String)
    fuel = Column(String)
    mileage = Column(Integer)
    transmission = Column(String)
    doors = Column(Integer)
    price = Column(Integer)
    selling_price = Column(Integer)
    location = Column(String)
    geolocation = Column(String)
    budget = Column(Integer)
    stage = Column(String)
    next_stage = Column(String)
    phone_number = Column(String)
    email = Column(String)
    first_message = Column(String)
    last_updated = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    features = Column(String)  # Comma-separated list of features
    car_suggestions = Column(String)  # Comma-separated list of suggested cars
    negotiation_state = Column(String)  # JSON string representing negotiation state
    


class ConversationHistory(Base):
    __tablename__ = "conversation_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String)
    user_message = Column(String)
    assistant_message = Column(String)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
