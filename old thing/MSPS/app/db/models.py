from sqlalchemy import Column, Integer, Float, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class SampleRecord(Base):
    __tablename__ = "samples"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    raw = Column(JSON)          # raw uploaded features
    ucs_pred = Column(Float)
    slope_prob = Column(Float)
    msi = Column(Float)
    msi_cat = Column(String)
    notes = Column(String, nullable=True)
