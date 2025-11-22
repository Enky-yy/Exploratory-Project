from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.models import Base
import os


DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./msps.db')
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith('sqlite') else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# create tables at import time
Base.metadata.create_all(bind=engine)




def get_db_session():
 return SessionLocal()