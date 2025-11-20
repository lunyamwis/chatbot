from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, DateTime, func


# Change DB URL here when switching databases
# SQLite:
DATABASE_URL = "sqlite:///chatbot.db"


engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


class ConversationMemory(Base):
    __tablename__ = "conversation_memory"

    user_id = Column(String, primary_key=True)
    make = Column(String)
    model = Column(String)
    drive = Column(String)
    body_type = Column(String)
    colour = Column(String)
    budget = Column(Integer)
    stage = Column(String)



class ConversationHistory(Base):
    __tablename__ = "conversation_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String)
    user_message = Column(String)
    assistant_message = Column(String)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
