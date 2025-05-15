from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Text

Base = declarative_base()

class EmotionLog(Base):
    __tablename__ = "emotionData"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(String(7), nullable=False)
    input = Column(Text, nullable=False)
    output = Column(String(20), nullable=False)
