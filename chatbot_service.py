from dotenv import load_dotenv
load_dotenv()

import os
import sqlite3
import pandas as pd
import re
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

# Check if google.generativeai is available
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
    
    # Read API key directly from .env file
    from dotenv import dotenv_values
    config = dotenv_values(".env")
    api_key = config.get("GOOGLE_API_KEY")
    
    if api_key:
        # Configure Genai Key
        genai.configure(api_key=api_key)
    else:
        print("API key not found in .env file!")
        GENAI_AVAILABLE = False
        
except ImportError:
    GENAI_AVAILABLE = False

app = FastAPI()

# Add CORS middleware to allow requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "https://*.streamlit.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: Optional[str] = None
    mode: str = "ask"
    custom_sql: Optional[str] = None

# Function to load Google Gemini Model and provide queries as response
def get_gemini_response(question, prompt):
    if not GENAI_AVAILABLE:
        return None
        
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content([prompt[0], question])
        return response.text
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return None

# Function to retrieve query from the database
def read_sql_query(sql, db):
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [description[0] for description in cur.description]
        conn.close()
        return rows, columns
    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
        return None, None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None, None

# Function to clean and validate SQL query
def clean_sql_query(sql_text):
    if not sql_text:
        return ""
        
    # Remove markdown code blocks if present
    cleaned_sql = re.sub(r'```sql\s*|\s*```', '', sql_text, flags=re.IGNORECASE).strip()
    
    # Ensure query ends with semicolon
    if not cleaned_sql.endswith(';'):
        cleaned_sql += ';'
        
    # Basic validation - check if it looks like a SQL query
    if not cleaned_sql.lower().startswith(('select', 'insert', 'update', 'delete', 'with')):
        print("Generated query doesn't appear to be a standard SELECT statement.")
    
    return cleaned_sql

# Define Your Prompt with updated schema and clearer instructions
prompt = [
    """
You are an expert in converting natural language questions into valid SQL queries for a SQLite database named mydatabase.db.

Database Schema:
Students(student_id, roll_number, full_name, dob, gender, blood_group, nationality, address, phone_number, email)
Guardians(guardian_id, student_id, guardian_name, relationship, guardian_phone, emergency_contact)
AcademicRecords(record_id, student_id, department, session_year, last_cgpa)
SemesterResults(result_id, student_id, semester_number, cgpa, attendance_percentage)
FaceEncodes(encode_id, student_id, image_path, face_encoding, created_at)
Faculty(faculty_id, name, username, password_hash)

Instructions:
- Your output must be ONLY the SQL query. No extra text, explanations, or formatting.
- Do NOT use code blocks (e.g., ```sql).
- Always use aliases for tables in JOINs (e.g., `Students s`).
- Connect tables using `student_id` for joins.
- Use `SELECT` queries only. Do not generate any other SQL commands.
- The query MUST end with a semicolon.

Examples:
- Question: How many students are there?
- SQL: SELECT COUNT(*) FROM Students;
- Question: What are the names of students in the Data Science department?
- SQL: SELECT T1.full_name FROM Students AS T1 JOIN AcademicRecords AS T2 ON T1.student_id = T2.student_id WHERE T2.department = 'Data Science';
- Question: Show students with a CGPA greater than 8.5 in semester 1.
- SQL: SELECT T1.full_name FROM Students AS T1 JOIN SemesterResults AS T2 ON T1.student_id = T2.student_id WHERE T2.semester_number = 1 AND T2.cgpa > 8.5;
    """
]

@app.post("/query")
async def process_query(request: QueryRequest):
    data_base = 'D:/DESKTOP/VisionID/students_data/mydatabase1.db'
    
    if request.mode == "ask" and request.question and GENAI_AVAILABLE:
        response = get_gemini_response(request.question, prompt)
        
        if response:
            # Clean the SQL response
            cleaned_sql = clean_sql_query(response)
            
            # Execute the query
            results, columns = read_sql_query(cleaned_sql, data_base)
            
            if results is not None:
                # Create a DataFrame for better display
                df = pd.DataFrame(results, columns=columns)
                return {
                    "sql": cleaned_sql,
                    "results": df.to_dict(orient='records'),
                    "columns": columns,
                    "row_count": len(results)
                }
            else:
                raise HTTPException(status_code=500, detail="Could not execute the generated SQL query.")
        else:
            raise HTTPException(status_code=500, detail="Could not generate a SQL query. Please try a different question.")
    
    elif request.mode == "direct" and request.custom_sql:
        # Clean the custom SQL input
        cleaned_sql = clean_sql_query(request.custom_sql)
        results, columns = read_sql_query(cleaned_sql, data_base)
        
        if results is not None:
            # Create a DataFrame for better display
            df = pd.DataFrame(results, columns=columns)
            return {
                "sql": cleaned_sql,
                "results": df.to_dict(orient='records'),
                "columns": columns,
                "row_count": len(results)
            }
        else:
            raise HTTPException(status_code=500, detail="Could not execute the SQL query.")
    else:
        raise HTTPException(status_code=400, detail="Invalid request parameters.")

@app.get("/health")
async def health_check():
    return {"status": "ok", "genai_available": GENAI_AVAILABLE}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)