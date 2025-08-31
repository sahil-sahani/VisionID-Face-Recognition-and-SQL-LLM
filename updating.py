import sqlite3
import os
import shutil
import face_recognition
import pickle
from datetime import datetime

def create_connection():
    """Create a database connection to the SQLite database"""
    conn = None
    try:
        conn = sqlite3.connect('D:\DESKTOP\VisionID\students_data\mydatabase1.db')
        print(f"Connected to SQLite database (version {sqlite3.version})")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
    return conn

def initialize_database(conn):
    """Initialize the database with required tables if they don't exist"""
    try:
        cursor = conn.cursor()
        
        # Create Students table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Students (
                student_id INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_number TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                dob DATE,
                gender TEXT CHECK(gender IN ('Male', 'Female', 'Other')),
                blood_group TEXT,
                nationality TEXT,
                address TEXT,
                phone_number TEXT,
                email TEXT UNIQUE
            )
        ''')
        
        # Create SemesterResults table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS SemesterResults (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                semester_number INTEGER CHECK (semester_number > 0),
                cgpa REAL,
                attendance_percentage REAL,
                FOREIGN KEY (student_id) REFERENCES Students (student_id)
            )
        ''')
        
        # Create FaceEncodes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS FaceEncodes (
                encode_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                image_path TEXT,
                face_encoding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES Students (student_id)
            )
        ''')
        
        conn.commit()
        print("Database initialized successfully")
    except sqlite3.Error as e:
        print(f"Error initializing database: {e}")

def get_student_by_id(conn, student_id):
    """Retrieve student by ID"""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Students WHERE student_id = ?", (student_id,))
        student = cursor.fetchone()
        return student
    except sqlite3.Error as e:
        print(f"Error retrieving student: {e}")
        return None

def generate_face_encoding(image_path):
    """Generate face encoding from an image"""
    try:
        # Load the image
        image = face_recognition.load_image_file(image_path)
        
        # Find face locations in the image
        face_locations = face_recognition.face_locations(image)
        
        if len(face_locations) == 0:
            print("No face detected in the image.")
            return None
        
        # Generate face encodings
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if len(face_encodings) == 0:
            print("Could not generate face encoding.")
            return None
        
        # Return the first face encoding (assuming one face per image)
        return face_encodings[0]
    except Exception as e:
        print(f"Error generating face encoding: {e}")
        return None

def save_face_encoding(conn, student_id, image_path, face_encoding):
    """Save face encoding to database"""
    try:
        cursor = conn.cursor()
        
        # Serialize the face encoding
        encoding_blob = pickle.dumps(face_encoding)
        
        # Check if encoding already exists for this student
        cursor.execute(
            "SELECT encode_id FROM FaceEncodes WHERE student_id = ?",
            (student_id,)
        )
        existing_encoding = cursor.fetchone()
        
        if existing_encoding:
            # Update existing record
            cursor.execute(
                "UPDATE FaceEncodes SET image_path = ?, face_encoding = ?, created_at = ? WHERE student_id = ?",
                (image_path, encoding_blob, datetime.now(), student_id)
            )
        else:
            # Insert new record
            cursor.execute(
                "INSERT INTO FaceEncodes (student_id, image_path, face_encoding) VALUES (?, ?, ?)",
                (student_id, image_path, encoding_blob)
            )
        
        conn.commit()
        print("Face encoding saved to database successfully!")
        return True
    except sqlite3.Error as e:
        conn.rollback()
        print(f"Error saving face encoding: {e}")
        return False
    except Exception as e:
        conn.rollback()
        print(f"Error processing face encoding: {e}")
        return False

def update_student_info(conn, student_id, full_name, cgpa, image_path):
    """Update student information including image and face encoding"""
    try:
        cursor = conn.cursor()
        
        # Update student name
        cursor.execute(
            "UPDATE Students SET full_name = ? WHERE student_id = ?",
            (full_name, student_id)
        )
        
        # Get the latest semester for the student
        cursor.execute(
            "SELECT semester_number FROM SemesterResults WHERE student_id = ? ORDER BY semester_number DESC LIMIT 1",
            (student_id,)
        )
        latest_semester = cursor.fetchone()
        
        if latest_semester:
            # Update the latest CGPA
            cursor.execute(
                "UPDATE SemesterResults SET cgpa = ? WHERE student_id = ? AND semester_number = ?",
                (cgpa, student_id, latest_semester[0])
            )
        else:
            # If no semester record exists, create one
            cursor.execute(
                "INSERT INTO SemesterResults (student_id, semester_number, cgpa, attendance_percentage) VALUES (?, 1, ?, 0.0)",
                (student_id, cgpa)
            )
        
        # Handle image update and face encoding
        if image_path and os.path.exists(image_path):
            # Get student roll number for image naming
            cursor.execute("SELECT roll_number FROM Students WHERE student_id = ?", (student_id,))
            roll_number = cursor.fetchone()[0]
            
            # Create images directory if it doesn't exist
            images_dir = "students_data\stds_images"
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            
            # Copy image to destination with roll_number.jpg name
            dest_path = os.path.join(images_dir, f"{roll_number}.jpg")
            shutil.copy2(image_path, dest_path)
            print(f"Image saved to: {dest_path}")
            
            # Generate and save face encoding
            face_encoding = generate_face_encoding(dest_path)
            if face_encoding is not None:
                save_face_encoding(conn, student_id, dest_path, face_encoding)
        
        conn.commit()
        print("Student information updated successfully!")
        return True
    except sqlite3.Error as e:
        conn.rollback()
        print(f"Error updating student information: {e}")
        return False
    except Exception as e:
        conn.rollback()
        print(f"Error handling image: {e}")
        return False

def main():
    # Create database connection
    conn = create_connection()
    if conn is None:
        return
    
    # Initialize database
    initialize_database(conn)
    
    while True:
        print("\n=== Student Information Update System ===")
        print("1. Update student information")
        print("2. Exit")
        
        choice = input("Enter your choice (1-2): ").strip()
        
        if choice == "1":
            try:
                student_id = int(input("Enter student ID: "))
                
                # Check if student exists
                student = get_student_by_id(conn, student_id)
                if not student:
                    print(f"Student with ID {student_id} not found.")
                    continue
                
                print(f"\nCurrent student information:")
                print(f"ID: {student[0]}, Roll Number: {student[1]}, Name: {student[2]}")
                
                # Get updated information
                new_name = input("Enter new full name (press Enter to keep current): ").strip()
                if not new_name:
                    new_name = student[2]  # Keep current name
                
                # Get current CGPA
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT cgpa FROM SemesterResults WHERE student_id = ? ORDER BY semester_number DESC LIMIT 1",
                    (student_id,)
                )
                current_cgpa = cursor.fetchone()
                current_cgpa = current_cgpa[0] if current_cgpa else 0.0
                
                print(f"Current CGPA: {current_cgpa}")
                try:
                    new_cgpa = float(input("Enter new CGPA (press Enter to keep current): ").strip() or current_cgpa)
                except ValueError:
                    print("Invalid CGPA value. Keeping current CGPA.")
                    new_cgpa = current_cgpa
                
                new_image_path = input("Enter path to new image (press Enter to skip): ").strip()
                
                # Update student information
                success = update_student_info(conn, student_id, new_name, new_cgpa, new_image_path)
                if success:
                    print("Update completed successfully!")
                else:
                    print("Update failed.")
                    
            except ValueError:
                print("Please enter a valid student ID (integer).")
            except Exception as e:
                print(f"An error occurred: {e}")
                
        elif choice == "2":
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
    
    # Close database connection
    conn.close()

if __name__ == "__main__":
    main()