from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from decouple import config

DATABASE_URL = config("DATABASE_URL")

# MySQL-specific engine configuration
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Enables pessimistic disconnect handling
    pool_recycle=300,    # Recycle connections every 5 minutes
    echo=False           # Set to True for debugging SQL queries
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()