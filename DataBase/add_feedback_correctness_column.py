"""
Simple migration: Add feedback_correctness column to user_choices table
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URI = os.getenv("AZURE_DATABASE_URI", "sqlite:///test.db")

try:
    engine = create_engine(DATABASE_URI)
    
    with engine.connect() as conn:
        print("Adding feedback_correctness column...")
        conn.execute(text("ALTER TABLE user_choices ADD COLUMN feedback_correctness INTEGER NULL"))
        conn.commit()
        print("✓ Column added successfully")
        
except Exception as e:
    print(f"Error: {e}")
    if "Duplicate column" in str(e) or "already exists" in str(e):
        print("✓ Column already exists")
