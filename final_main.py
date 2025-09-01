import streamlit as st
import sqlite3
import face_recognition
import numpy as np
import pickle
from PIL import Image
import pandas as pd
from datetime import datetime
import os
import json
import base64
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from sklearn.cluster import KMeans
import plotly.express as px
import re
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if user is logged in
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login form
if not st.session_state.logged_in:
    # Set page config for login page
    st.set_page_config(page_title="VisionID", page_icon="🎓", layout="centered")
    
    # Title in the middle
    st.markdown(
        "<h1 style='text-align: center; color:black;'>VisionID</h1>", 
        unsafe_allow_html=True
    )

    # Login form
    with st.form("login_form"):
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        if username == "admin" and password == "admin":
            st.session_state.logged_in = True
            st.success("Login successful! Welcome Admin ✅")
            st.rerun()
        else:
            st.error("Invalid Username or Password ❌")
    
    # Stop execution here if not logged in
    st.stop()

# If logged in, continue with the main application
# Set up the page configuration for the main app
st.set_page_config(
    page_title="VisionID",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS ----
st.markdown("""
    <style>
    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #f5f7fa;
    }
    /* Sidebar title */
    .css-1d391kg {
        font-size: 22px;
        font-weight: 700;
        color: #2c2c2c;
    }
    /* Buttons */
    .stButton > button {
        width: 100%;
        height: 45px;
        border-radius: 8px;
        border: 1px solid #ddd;
        background-color: white;
        color: black;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #e6f0ff;
        border: 1px solid #3399ff;
    }
    /* Dropdown */
    div[data-baseweb="select"] > div {
        border: 1px solid #ddd !important;
        border-radius: 8px;
    }
    /* Vision Search specific styles */
    .small-image {
        max-width: 300px;
        margin: 0 auto;
        display: block;
    }
    .image-container {
        text-align: center;
        margin-bottom: 20px;
    }
    .unknown-person {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 20px;
        margin: 20px 0;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Data Loading Functions ----
def get_db_connection():
    conn = sqlite3.connect('D:/DESKTOP/VisionID/students_data/mydatabase1.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

@st.cache_data
def load_data():
    with get_db_connection() as conn:
        query = """
        SELECT
            s.student_id, s.roll_number, s.full_name, s.dob, s.gender,
            s.blood_group, s.nationality, s.address, s.phone_number, s.email,
            ar.department, ar.session_year, ar.last_cgpa,
            g.guardian_name, g.relationship, g.guardian_phone, g.emergency_contact
        FROM Students s
        LEFT JOIN AcademicRecords ar ON s.student_id = ar.student_id
        LEFT JOIN Guardians g ON s.student_id = g.student_id
        """
        students_df = pd.read_sql_query(query, conn)
        semester_query = """
        SELECT
            sr.student_id, sr.semester_number, sr.cgpa, sr.attendance_percentage,
            ar.department
        FROM SemesterResults sr
        JOIN AcademicRecords ar ON sr.student_id = ar.student_id
        """
        semester_df = pd.read_sql_query(semester_query, conn)
    return students_df, semester_df

def get_student_data(student_id):
    with get_db_connection() as conn:
        student_query = "SELECT * FROM Students WHERE student_id = ?"
        student_data = conn.execute(student_query, (student_id,)).fetchone()
        guardian_query = "SELECT * FROM Guardians WHERE student_id = ?"
        guardian_data = conn.execute(guardian_query, (student_id,)).fetchall()
        academic_query = "SELECT * FROM AcademicRecords WHERE student_id = ?"
        academic_data = conn.execute(academic_query, (student_id,)).fetchone()
        semester_query = "SELECT * FROM SemesterResults WHERE student_id = ? ORDER BY semester_number"
        semester_data = conn.execute(semester_query, (student_id,)).fetchall()
    return student_data, guardian_data, academic_data, semester_data

def plot_student_trends_and_comparison(student_id, semester_data):
    if not semester_data:
        return None
    df_cgpa = pd.DataFrame(
        semester_data,
        columns=['result_id', 'student_id', 'semester_number', 'cgpa', 'attendance_percentage']
    )
    student_data = df_cgpa[df_cgpa['student_id'] == student_id]
    if student_data.empty:
        return None
    with get_db_connection() as conn:
        dept_avg_query = """
        SELECT sr.semester_number, AVG(sr.cgpa) as cgpa
        FROM SemesterResults sr
        JOIN AcademicRecords ar ON sr.student_id = ar.student_id
        GROUP BY sr.semester_number
        """
        dept_avg = pd.read_sql_query(dept_avg_query, conn)
    merged = pd.merge(
        student_data,
        dept_avg,
        on='semester_number',
        suffixes=('_student', '_dept')
    )
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("CGPA & Attendance Trend", "Student vs Dept Avg CGPA per Semester")
    )
    # Trend line plot (CGPA and Attendance)
    fig.add_trace(
        go.Scatter(
            x=student_data['semester_number'],
            y=student_data['cgpa'],
            mode='lines+markers',
            name='Student CGPA'
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=student_data['semester_number'],
            y=student_data['attendance_percentage'],
            mode='lines+markers',
            name='Student Attendance',
            line=dict(dash='dot')
        ),
        row=1,
        col=1
    )
    # Bar chart (only CGPA)
    fig.add_trace(
        go.Bar(
            x=merged['semester_number'],
            y=merged['cgpa_student'],
            name='Student CGPA'
        ),
        row=1,
        col=2
    )
    fig.add_trace(
        go.Bar(
            x=merged['semester_number'],
            y=merged['cgpa_dept'],
            name='Dept Avg CGPA'
        ),
        row=1,
        col=2
    )
    fig.update_layout(
        barmode='group',
        title_text='Student Performance vs Department Average',
        width=1000,
        height=500,
        showlegend=True
    )
    fig.update_xaxes(title_text='Semester', row=1, col=1)
    fig.update_xaxes(title_text='Semester', row=1, col=2)
    fig.update_yaxes(title_text='Value', row=1, col=1)
    fig.update_yaxes(title_text='CGPA', row=1, col=2)
    return fig

# ---- Vision Search Functions ----
def decode_face_encoding(encoding_data):
    try:
        return pickle.loads(encoding_data)
    except:
        try:
            if isinstance(encoding_data, bytes):
                try:
                    decoded_data = base64.b64decode(encoding_data)
                    return np.frombuffer(decoded_data, dtype=np.float64)
                except:
                    return np.frombuffer(encoding_data, dtype=np.float64)
        except:
            try:
                if isinstance(encoding_data, str):
                    encoding_list = json.loads(encoding_data)
                    return np.array(encoding_list, dtype=np.float64)
                elif isinstance(encoding_data, bytes):
                    encoding_list = json.loads(encoding_data.decode('utf-8'))
                    return np.array(encoding_list, dtype=np.float64)
            except:
                try:
                    if isinstance(encoding_data, bytes):
                        encoding_str = encoding_data.decode('utf-8')
                    else:
                        encoding_str = str(encoding_data)
                    clean_str = encoding_str.replace('[', '').replace(']', '')
                    numbers = [float(x) for x in clean_str.split() if x]
                    return np.array(numbers, dtype=np.float64)
                except Exception as e:
                    st.error(f"Could not decode face encoding: {e}")
                    return None
    return None

def get_all_face_encodings():
    conn = get_db_connection()
    if conn is None:
        return [], [], []
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT encode_id, student_id, image_path, face_encoding FROM FaceEncodes")
        rows = cursor.fetchall()
        known_encodings = []
        known_ids = []
        known_image_paths = []
        problematic_encodings = []
        for row in rows:
            student_id = row['student_id']
            encode_id = row['encode_id']
            image_path = row['image_path']
            encoding_data = row['face_encoding']
            face_encoding = decode_face_encoding(encoding_data)
            if face_encoding is not None and len(face_encoding) == 128:
                known_encodings.append(face_encoding)
                known_ids.append(student_id)
                known_image_paths.append(image_path)
            else:
                problematic_encodings.append(encode_id)
        return known_encodings, known_ids, known_image_paths
    except sqlite3.Error as e:
        st.error(f"Error retrieving face encodings: {e}")
        return [], [], []
    finally:
        conn.close()

def process_image(image):
    image.thumbnail((400, 400))
    rgb_image = np.array(image.convert('RGB'))
    face_encodings = face_recognition.face_encodings(rgb_image)
    if len(face_encodings) == 0:
        st.error("No faces detected in the image. Please try another image.")
        return None
    return face_encodings[0]

def find_match(unknown_encoding, known_encodings, known_ids, known_image_paths, tolerance=0.5):
    if not known_encodings:
        return None, None, None, None
    matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=tolerance)
    face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
    if len(face_distances) > 0:
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            return known_ids[best_match_index], known_encodings[best_match_index], face_distances[best_match_index], known_image_paths[best_match_index]
    return None, None, None, None

def get_roll_number(student_id):
    conn = get_db_connection()
    if conn is None:
        return None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT roll_number FROM Students WHERE student_id = ?", (student_id,))
        result = cursor.fetchone()
        return result['roll_number'] if result else None
    except sqlite3.Error as e:
        st.error(f"Error retrieving roll number: {e}")
        return None
    finally:
        conn.close()

def show_student_dashboard_vision(student_id):
    student_data, guardian_data, academic_data, semester_data = get_student_data(student_id)
    if student_data is None:
        st.error("Student not found!")
        return False
    roll_number = student_data['roll_number']
    left_col, right_col = st.columns([1, 2])
    with left_col:
        image_path = f"D:/DESKTOP/VisionID/students_data/stds_images/{roll_number}.jpg"
        if os.path.exists(image_path):
            image = Image.open(image_path)
            image.thumbnail((200, 200))
            st.image(image, width=150)
        else:
            st.warning("Student image not found")
        st.write(f"**Full Name:** {student_data['full_name']}")
        st.write(f"**Roll No:** {roll_number}")
        if academic_data:
            st.write(f"**Branch:** {academic_data['department']}")
            st.write(f"**Session:** {academic_data['session_year']}")
            st.write(f"**Last CGPA:** {academic_data['last_cgpa']}")
    with right_col:
        tab1, tab2, tab3 = st.tabs(["Personal Details", "Guardian Details", "Academic Performance"])
        with tab1:
            personal_details = {
                "Date of Birth": student_data['dob'],
                "Gender": student_data['gender'],
                "Blood Group": student_data['blood_group'],
                "Nationality": student_data['nationality'],
                "Phone": student_data['phone_number'],
                "Email": student_data['email'],
                "Address": student_data['address']
            }
            for key, value in personal_details.items():
                if value:
                    st.write(f"**{key}:** {value}")
        with tab2:
            if guardian_data:
                for i, guardian in enumerate(guardian_data):
                    st.write(f"**Name:** {guardian['guardian_name']}")
                    st.write(f"**Relationship:** {guardian['relationship']}")
                    st.write(f"**Phone:** {guardian['guardian_phone']}")
                    if guardian['emergency_contact']:
                        st.write(f"**Emergency Contact:** {guardian['emergency_contact']}")
                    if i < len(guardian_data) - 1:
                        st.divider()
            else:
                st.warning("No guardian information found for this student.")
        with tab3:
            if semester_data:
                st.subheader("Semester-wise Performance")
                semester_df = pd.DataFrame(semester_data, columns=['result_id', 'student_id', 'semester_number', 'cgpa', 'attendance_percentage'])
                semester_df = semester_df[['semester_number', 'cgpa', 'attendance_percentage']]
                semester_df.rename(columns={
                    'semester_number': 'Semester',
                    'cgpa': 'CGPA',
                    'attendance_percentage': 'Attendance %'
                }, inplace=True)
                st.dataframe(semester_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No semester data available for this student.")
    st.markdown("---")
    st.header("Academic Performance Charts")
    if semester_data:
        fig = plot_student_trends_and_comparison(student_id, semester_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No semester data available for performance charts.")
    return True

def show_face_recognition_interface():
    st.title("Student Information Via Image")
    conn = get_db_connection()
    if conn is None:
        st.error("Cannot connect to the database. Please check the database path.")
        return
    conn.close()
    with st.spinner("Loading face encodings from database..."):
        known_encodings, known_ids, known_image_paths = get_all_face_encodings()
    if not known_encodings:
        st.warning("No valid face encodings found in the database. Please add students first.")
        return
    input_method = st.radio("Select Input Method", ("Upload Image", "Capture from Camera"))
    unknown_encoding = None
    captured_image = None
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload Student Image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image.thumbnail((300, 300))
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image", width=250)
            st.markdown('</div>', unsafe_allow_html=True)
            unknown_encoding = process_image(image)
            captured_image = image
    else:
        picture = st.camera_input("Capture Student Image")
        if picture:
            image = Image.open(picture)
            image.thumbnail((300, 300))
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, caption="Captured Image", width=250)
            st.markdown('</div>', unsafe_allow_html=True)
            unknown_encoding = process_image(image)
            captured_image = image
    if unknown_encoding is not None and not st.session_state.get('confirmation', False) and not st.session_state.get('processing_complete', False):
        with st.spinner("Matching face..."):
            student_id, matched_encoding, distance, matched_image_path = find_match(unknown_encoding, known_encodings, known_ids, known_image_paths)
            if student_id:
                st.session_state.matched_student = {
                    'student_id': student_id,
                    'encoding': matched_encoding,
                    'distance': distance,
                    'image': captured_image,
                    'original_image_path': matched_image_path
                }
                student_data = get_student_data(student_id)[0]
                full_name = student_data['full_name'] if student_data else "N/A"
                roll_number = student_data['roll_number'] if student_data else None
                correct_image_path = f"D:/DESKTOP/VisionID/students_data/stds_images/{roll_number}.jpg" if roll_number else None
                match_percentage = max(0, 100 * (1 - distance / 0.6))
                st.success(
                    f"✅ Potential match found: **{full_name}** (Roll No: {roll_number})\n\n"
                    f"🔹 Match Confidence: **{match_percentage:.2f}%** "
                    f"(distance: {distance:.4f})"
                )
                col1, col2 = st.columns(2)
                with col1:
                    st.image(captured_image, caption="Uploaded / Captured Image", width=200)
                with col2:
                    if correct_image_path and os.path.exists(correct_image_path):
                        st.image(correct_image_path, caption="Original Student Image", width=200)
                    else:
                        st.warning("Original student image not found.")
                st.markdown(
                    "<h4 style='text-align:center;'>Is this the person you are looking for?</h4>",
                    unsafe_allow_html=True
                )
                col_yes, col_no = st.columns([1, 1])
                with col_yes:
                    if st.button("✅ Yes, Confirm"):
                        st.session_state.confirmation = True
                        st.session_state.recognition_complete = True
                        st.session_state.processing_complete = True
                        st.rerun()
                with col_no:
                    if st.button("❌ No, Try Again"):
                        st.session_state.matched_student = None
                        st.session_state.confirmation = False
                        st.session_state.processing_complete = True
                        st.rerun()
            else:
                st.session_state.unknown_person = True
                st.session_state.processing_complete = True
                st.markdown(
                    """
                    <div class="unknown-person">
                        <h3>❌ Person Not Recognized</h3>
                        <p>This person is not in our database. Please try again or contact the administrator.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.image(captured_image, caption="Unknown Person", width=250)
                if st.button("🔄 Try Again"):
                    st.session_state.unknown_person = False
                    st.session_state.processing_complete = False
                    st.rerun()

def vision_search_page():
    if 'matched_student' not in st.session_state:
        st.session_state.matched_student = None
    if 'confirmation' not in st.session_state:
        st.session_state.confirmation = False
    if 'recognition_complete' not in st.session_state:
        st.session_state.recognition_complete = False
    if 'scrolled_to_top' not in st.session_state:
        st.session_state.scrolled_to_top = False
    if 'unknown_person' not in st.session_state:
        st.session_state.unknown_person = False
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if st.session_state.recognition_complete and st.session_state.matched_student:
        if st.button("← Back to Face Recognition"):
            st.session_state.matched_student = None
            st.session_state.confirmation = False
            st.session_state.recognition_complete = False
            st.session_state.processing_complete = False
            st.session_state.scrolled_to_top = False
            st.session_state.unknown_person = False
            st.rerun()
        if not st.session_state.scrolled_to_top:
            st.markdown(
                """
                <script>
                    window.scrollTo(0, 0);
                </script>
                """,
                unsafe_allow_html=True
            )
            st.session_state.scrolled_to_top = True
        show_student_dashboard_vision(st.session_state.matched_student['student_id'])
    else:
        show_face_recognition_interface()

# Load the data
students_df, semester_df = load_data()

# ---- Sidebar ----
st.markdown(
    """
    <style>
    .sidebar-title {
        font-size: 2.5em !important;
        font-weight: bold !important;
        text-align: center !important;
        margin-bottom: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown('<div class="sidebar-title">VisionID</div>', unsafe_allow_html=True)

    if st.button("📊 Dashboard"):
        st.session_state["page"] = "dashboard"
    st.markdown("#### Department Filter")
    departments = ["All"] + list(students_df['department'].dropna().unique())
    dept_filter = st.selectbox("Select Department", departments, key="dept_filter")
    if st.button("👁 Vision Search"):
        st.session_state["page"] = "vision_search"
    if st.button("💬 DataQuery Assistant"):
        st.session_state["page"] = "chatbot"
        
    # Add logout button
    st.markdown("---")
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

if "page" not in st.session_state:
    st.session_state["page"] = "dashboard"

# ---- Main Content ----
if st.session_state["page"] == "dashboard":
    if dept_filter == "All":
        filtered_students = students_df
        filtered_semester = semester_df
        title = "Institutional Data Overview"
    else:
        filtered_students = students_df[students_df['department'] == dept_filter]
        filtered_semester = semester_df[semester_df['department'] == dept_filter]
        title = f"{dept_filter} Department Dashboard"
    st.title(title)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_students = filtered_students['student_id'].nunique()
        st.metric("Total Students", total_students)
    with col2:
        avg_cgpa = filtered_students['last_cgpa'].mean()
        st.metric("Average CGPA", f"{avg_cgpa:.2f}" if not pd.isna(avg_cgpa) else "N/A")
    with col3:
        avg_attendance = filtered_semester['attendance_percentage'].mean()
        st.metric("Average Attendance", f"{avg_attendance:.1f}%" if not pd.isna(avg_attendance) else "N/A")
    with col4:
        if dept_filter == "All":
            dept_count = filtered_students['department'].nunique()
            st.metric("Departments", dept_count)
        else:
            active_semesters = filtered_semester['semester_number'].nunique()
            st.metric("Active Semesters", active_semesters)
    st.markdown("---")
    if dept_filter == "All":
        col1, col2 = st.columns((2, 1))
        with col1:
            st.subheader("Student Distribution by Department and Gender")
            sunburst_data = filtered_students[['department', 'gender']].copy()
            sunburst_data = sunburst_data.dropna()
            sunburst_data = sunburst_data.groupby(['department', 'gender']).size().reset_index(name='count')
            if not sunburst_data.empty:
                fig_sunburst = px.sunburst(sunburst_data, path=['department', 'gender'], values='count',
                                           color='department',
                                           title='Interactive Student Distribution')
                fig_sunburst.update_layout(margin=dict(t=50, l=0, r=0, b=0))
                st.plotly_chart(fig_sunburst, use_container_width=True)
            else:
                st.info("No data available for sunburst chart.")
        with col2:
            st.subheader("Overall Gender Distribution")
            gender_counts = filtered_students['gender'].value_counts()
            if not gender_counts.empty:
                fig_pie = px.pie(values=gender_counts.values, names=gender_counts.index,
                                 title='Gender Breakdown', hole=0.4,
                                 color_discrete_sequence=px.colors.sequential.RdBu)
                fig_pie.update_traces(textinfo='percent+label', pull=[0.05, 0])
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No gender data available.")
        st.markdown("---")
        st.subheader("Department Performance Comparison")
        col1, col2 = st.columns(2)
        with col1:
            avg_cgpa_dept = filtered_students.groupby('department')['last_cgpa'].mean().reset_index()
            avg_cgpa_dept = avg_cgpa_dept.dropna()
            if not avg_cgpa_dept.empty:
                avg_cgpa_dept = avg_cgpa_dept.sort_values('last_cgpa', ascending=False)
                fig_cgpa_bar = px.bar(
                    avg_cgpa_dept,
                    x='department',
                    y='last_cgpa',
                    title='Average CGPA per Department',
                    text_auto='.2f',
                    labels={'last_cgpa': 'Average CGPA', 'department': 'Department'}
                )
                fig_cgpa_bar.update_traces(marker_color='#007bff', textposition='inside')
                st.plotly_chart(fig_cgpa_bar, use_container_width=True)
            else:
                st.info("No CGPA data available.")
        with col2:
            avg_att_dept = filtered_semester.groupby('department')['attendance_percentage'].mean().reset_index()
            avg_att_dept = avg_att_dept.dropna()
            if not avg_att_dept.empty:
                avg_att_dept = avg_att_dept.sort_values('attendance_percentage', ascending=False)
                fig_att_bar = px.bar(
                    avg_att_dept,
                    x='department',
                    y='attendance_percentage',
                    title='Average Attendance per Department',
                    text_auto='.1f',
                    labels={'attendance_percentage': 'Avg Attendance (%)', 'department': 'Department'}
                )
                fig_att_bar.update_traces(marker_color='#28a745', textposition='inside')
                st.plotly_chart(fig_att_bar, use_container_width=True)
            else:
                st.info("No attendance data available.")
        cluster_data = filtered_students[['student_id', 'last_cgpa']].copy()
        avg_att = filtered_semester.groupby('student_id')['attendance_percentage'].mean().reset_index()
        cluster_data = cluster_data.merge(avg_att, on='student_id', how='inner')
        cluster_data = cluster_data.dropna()
        if len(cluster_data) > 0:
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            cluster_data['cluster'] = kmeans.fit_predict(cluster_data[['last_cgpa', 'attendance_percentage']])
            fig_cluster = px.scatter(cluster_data, x='last_cgpa', y='attendance_percentage', color='cluster',
                                    title='Student Clustering by CGPA and Attendance',
                                    labels={'last_cgpa': 'CGPA', 'attendance_percentage': 'Attendance Percentage'})
            st.plotly_chart(fig_cluster, use_container_width=True)
            cluster_desc = """
            **Cluster Interpretation:**
            - Cluster 0: Moderate CGPA, Moderate Attendance
            - Cluster 1: High CGPA, High Attendance
            - Cluster 2: Low CGPA, Low Attendance
            - Cluster 3: High CGPA, Low Attendance
            """
            st.markdown(cluster_desc)
        else:
            st.warning("Insufficient data for clustering analysis.")
        st.subheader("All Students Details")
        student_list = filtered_students[['student_id', 'roll_number', 'full_name', 'gender', 'department', 'last_cgpa']].copy()
        student_attendance = filtered_semester.groupby('student_id')['attendance_percentage'].mean().reset_index()
        student_list = student_list.merge(student_attendance, on='student_id', how='left')
        guardian_info = filtered_students.groupby('student_id').agg({
            'guardian_name': 'first',
            'relationship': 'first',
            'guardian_phone': 'first'
        }).reset_index()
        student_list = student_list.merge(guardian_info, on='student_id', how='left')
        student_list.columns = [
            'ID', 'Roll Number', 'Name', 'Gender', 'Department', 'CGPA',
            'Avg Attendance', 'Guardian Name', 'Relationship', 'Guardian Phone'
        ]
        st.dataframe(student_list, use_container_width=True, hide_index=True)
        st.subheader("Individual Student Profile")
        student_options = [f"{row['student_id']} - {row['full_name']}" for _, row in filtered_students[['student_id', 'full_name']].drop_duplicates().iterrows()]
        selected_student = st.selectbox("Select a Student", ["Select a student"] + student_options)
        if selected_student != "Select a student":
            student_id = int(selected_student.split(" - ")[0])
            student_data, guardian_data, academic_data, semester_data = get_student_data(student_id)
            if student_data:
                left_col, right_col = st.columns([1, 2])
                with left_col:
                    roll_no = student_data['roll_number']
                    image_path = f"D:/DESKTOP/VisionID/students_data/stds_images/{roll_no}.jpg"
                    if os.path.exists(image_path):
                        image = Image.open(image_path)
                        st.image(image, width=200)
                    else:
                        st.warning("Student image not found")
                    st.write(f"**Full Name:** {student_data['full_name']}")
                    st.write(f"**Roll No:** {roll_no}")
                    if academic_data:
                        st.write(f"**Branch:** {academic_data['department']}")
                        st.write(f"**Session:** {academic_data['session_year']}")
                        st.write(f"**Last CGPA:** {academic_data['last_cgpa']}")
                with right_col:
                    tab1, tab2, tab3 = st.tabs(["Personal Details", "Guardian Details", "Academic Performance"])
                    with tab1:
                        personal_details = {
                            "Date of Birth": student_data['dob'],
                            "Gender": student_data['gender'],
                            "Blood Group": student_data['blood_group'],
                            "Nationality": student_data['nationality'],
                            "Phone": student_data['phone_number'],
                            "Email": student_data['email'],
                            "Address": student_data['address']
                        }
                        for key, value in personal_details.items():
                            if value:
                                st.write(f"**{key}:** {value}")
                    with tab2:
                        if guardian_data:
                            for i, guardian in enumerate(guardian_data):
                                st.write(f"**Name:** {guardian['guardian_name']}")
                                st.write(f"**Relationship:** {guardian['relationship']}")
                                st.write(f"**Phone:** {guardian['guardian_phone']}")
                                if guardian['emergency_contact']:
                                    st.write(f"**Emergency Contact:** {guardian['emergency_contact']}")
                                if i < len(guardian_data) - 1:
                                    st.divider()
                        else:
                            st.warning("No guardian information found for this student.")
                    with tab3:
                        if semester_data:
                            st.subheader("Semester-wise Performance")
                            semester_df = pd.DataFrame(
                                semester_data,
                                columns=['result_id', 'student_id', 'semester_number', 'cgpa', 'attendance_percentage']
                            )
                            semester_df = semester_df[['semester_number', 'cgpa', 'attendance_percentage']]
                            semester_df.rename(
                                columns={
                                    'semester_number': 'Semester',
                                    'cgpa': 'CGPA',
                                    'attendance_percentage': 'Attendance %'
                                },
                                inplace=True
                            )
                            st.dataframe(semester_df, use_container_width=True, hide_index=True)
                        else:
                            st.warning("No semester data available for this student.")
                st.markdown("---")
                st.header("Academic Performance Charts")
                if semester_data:
                    fig = plot_student_trends_and_comparison(student_id, semester_data)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No semester data available for performance charts.")
                else:
                    st.warning("No semester data available.")
            else:
                st.error("Student not found!")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Semester-wise Average CGPA")
            if not filtered_semester.empty:
                sem_cgpa = filtered_semester.groupby('semester_number')['cgpa'].mean().reset_index()
                fig_sem_line = px.line(sem_cgpa, x='semester_number', y='cgpa',
                                       title='Average CGPA Trend Across Semesters',
                                       labels={'semester_number': 'Semester', 'cgpa': 'Average CGPA'},
                                       markers=True)
                fig_sem_line.update_layout(xaxis_dtick=1)
                st.plotly_chart(fig_sem_line, use_container_width=True)
            else:
                st.info("No semester data available for this department.")
        with col2:
            st.subheader("CGPA Distribution")
            if not filtered_students.empty and 'last_cgpa' in filtered_students.columns:
                filtered_students_cgpa = filtered_students[filtered_students['last_cgpa'].notna()]
                if not filtered_students_cgpa.empty:
                    fig_dist = px.histogram(filtered_students_cgpa, x='last_cgpa', nbins=20,
                                            title='Distribution of Final CGPA',
                                            marginal="box",
                                            labels={'last_cgpa': 'Final CGPA'},
                                            color_discrete_sequence=['#636EFA'])
                    st.plotly_chart(fig_dist, use_container_width=True)
                else:
                    st.info("No CGPA data available for this department.")
            else:
                st.info("No CGPA data available for this department.")
        st.markdown("---")
        ranking_choice = st.radio(
            "Select Ranking View",
            ('Top 5 Performers', 'Bottom 5 Performers (At-Risk)'),
            horizontal=True
        )
        if ranking_choice == 'Top 5 Performers':
            ranked_students = filtered_students.nlargest(5, 'last_cgpa')
            chart_title = "Top 5 Students by CGPA"
            marker_color = '#28a745'
        else:
            ranked_students = filtered_students.nsmallest(5, 'last_cgpa')
            chart_title = "Bottom 5 Students by CGPA"
            marker_color = '#dc3545'
        if not ranked_students.empty and 'last_cgpa' in ranked_students.columns:
            fig_ranking = px.bar(ranked_students, x='last_cgpa', y='full_name', orientation='h',
                                 title=chart_title, text_auto='.2f',
                                 labels={'last_cgpa': 'CGPA', 'full_name': 'Student Name'},
                                 hover_data={'full_name': False, 'last_cgpa': ':.2f'})
            fig_ranking.update_traces(marker_color=marker_color)
            fig_ranking.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_ranking, use_container_width=True)
        else:
            st.info("No student data available for ranking.")
        st.subheader("Student Details")
        if not filtered_students.empty:
            student_list = filtered_students[['student_id', 'roll_number', 'full_name', 'gender', 'last_cgpa']].copy()
            student_attendance = filtered_semester.groupby('student_id')['attendance_percentage'].mean().reset_index()
            student_list = student_list.merge(student_attendance, on='student_id', how='left')
            guardian_info = filtered_students.groupby('student_id').agg({
                'guardian_name': 'first',
                'relationship': 'first',
                'guardian_phone': 'first'
            }).reset_index()
            student_list = student_list.merge(guardian_info, on='student_id', how='left')
            student_list.columns = [
                'ID', 'Roll Number', 'Name', 'Gender', 'CGPA',
                'Avg Attendance', 'Guardian Name', 'Relationship', 'Guardian Phone'
            ]
            st.dataframe(student_list, use_container_width=True, hide_index=True)
        st.subheader("Individual Student Profile")
        student_options = [f"{row['student_id']} - {row['full_name']}" for _, row in filtered_students[['student_id', 'full_name']].drop_duplicates().iterrows()]
        selected_student = st.selectbox("Select a Student", ["Select a student"] + student_options, key="dept_student_select")
        if selected_student != "Select a student":
            student_id = int(selected_student.split(" - ")[0])
            student_data, guardian_data, academic_data, semester_data = get_student_data(student_id)
            if student_data:
                left_col, right_col = st.columns([1, 2])
                with left_col:
                    roll_no = student_data['roll_number']
                    image_path = f"D:/DESKTOP/VisionID/students_data/stds_images/{roll_no}.jpg"
                    if os.path.exists(image_path):
                        image = Image.open(image_path)
                        st.image(image, width=200)
                    else:
                        st.warning("Student image not found")
                    st.write(f"**Full Name:** {student_data['full_name']}")
                    st.write(f"**Roll No:** {roll_no}")
                    if academic_data:
                        st.write(f"**Branch:** {academic_data['department']}")
                        st.write(f"**Session:** {academic_data['session_year']}")
                        st.write(f"**Last CGPA:** {academic_data['last_cgpa']}")
                with right_col:
                    tab1, tab2, tab3 = st.tabs(["Personal Details", "Guardian Details", "Academic Performance"])
                    with tab1:
                        personal_details = {
                            "Date of Birth": student_data['dob'],
                            "Gender": student_data['gender'],
                            "Blood Group": student_data['blood_group'],
                            "Nationality": student_data['nationality'],
                            "Phone": student_data['phone_number'],
                            "Email": student_data['email'],
                            "Address": student_data['address']
                        }
                        for key, value in personal_details.items():
                            if value:
                                st.write(f"**{key}:** {value}")
                    with tab2:
                        if guardian_data:
                            for i, guardian in enumerate(guardian_data):
                                st.write(f"**Name:** {guardian['guardian_name']}")
                                st.write(f"**Relationship:** {guardian['relationship']}")
                                st.write(f"**Phone:** {guardian['guardian_phone']}")
                                if guardian['emergency_contact']:
                                    st.write(f"**Emergency Contact:** {guardian['emergency_contact']}")
                                if i < len(guardian_data) - 1:
                                    st.divider()
                        else:
                            st.warning("No guardian information found for this student.")
                    with tab3:
                        if semester_data:
                            st.subheader("Semester-wise Performance")
                            semester_df = pd.DataFrame(
                                semester_data,
                                columns=['result_id', 'student_id', 'semester_number', 'cgpa', 'attendance_percentage']
                            )
                            semester_df = semester_df[['semester_number', 'cgpa', 'attendance_percentage']]
                            semester_df.rename(
                                columns={
                                    'semester_number': 'Semester',
                                    'cgpa': 'CGPA',
                                    'attendance_percentage': 'Attendance %'
                                },
                                inplace=True
                            )
                            st.dataframe(semester_df, use_container_width=True, hide_index=True)
                        else:
                            st.warning("No semester data available for this student.")
                st.markdown("---")
                st.header("Academic Performance Charts")
                if semester_data:
                    fig = plot_student_trends_and_comparison(student_id, semester_data)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No semester data available for performance charts.")
                else:
                    st.warning("No semester data available.")
            else:
                st.error("Student not found!")
elif st.session_state["page"] == "vision_search":
    vision_search_page()
elif st.session_state["page"] == "chatbot":
    # ===== NEW CHATBOT IMPLEMENTATION =====
# OpenRouter API configuration
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_MODEL = "deepseek/deepseek-chat-v3.1:free"  # Using the free version of DeepSeek

    # Check if OpenRouter API key is available
    try:
        from dotenv import dotenv_values
        config = dotenv_values("D:/DESKTOP/VisionID/.env1")
        api_key = config.get("DEEPSEEK_API_KEY")
        
        if not api_key:
            st.error("OpenRouter API key not found in .env file!")
            OPENROUTER_AVAILABLE = False
        else:
            OPENROUTER_AVAILABLE = True
            
    except Exception as e:
        st.error(f"Error loading API key: {str(e)}")
        OPENROUTER_AVAILABLE = False

    # Function to call OpenRouter API
    def get_ai_response(question, prompt):
        if not OPENROUTER_AVAILABLE:
            return None
            
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501",  # Streamlit default URL
                "X-Title": "VisionQuery App",
            }
            
            # Format the messages for the API
            messages = [
                {"role": "system", "content": prompt[0]},
                {"role": "user", "content": question}
            ]
            
            payload = {
                "model": OPENROUTER_MODEL,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 1000,
            }
            
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Error calling OpenRouter API: {str(e)}")
            return None
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
            st.error(f"Database error: {str(e)}")
            return None, None
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return None, None

    # Function to clean and validate SQL query
    def clean_sql_query(sql_text):
        # Remove markdown code blocks if present
        cleaned_sql = re.sub(r'```sql\s*|\s*```', '', sql_text, flags=re.IGNORECASE).strip()
        
        # Ensure query ends with semicolon
        if not cleaned_sql.endswith(';'):
            cleaned_sql += ';'
            
        # Basic validation - check if it looks like a SQL query
        if not cleaned_sql.lower().startswith(('select', 'insert', 'update', 'delete', 'with')):
            st.warning("Generated query doesn't appear to be a standard SELECT statement. Proceed with caution.")
        
        return cleaned_sql

    # Function to display results
    def display_results(results, columns):
        st.subheader("Query Results")
        
        if len(results) > 0:
            # Create a DataFrame for better display
            df = pd.DataFrame(results, columns=columns)
            
            # Display the results in a table
            st.dataframe(df)
            
            # Show some statistics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", len(df))
            col2.metric("Columns", len(columns))
            
            if len(df) > 0:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    col3.metric("Sample Value", df[numeric_cols[0]].iloc[0])
            
            # Allow data export
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No results found for your query.")

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

    st.header("💬 DataQuery Assistant")

    # Show API status
    if OPENROUTER_AVAILABLE:
        st.success("OpenRouter API: Available")
    else:
        st.warning("OpenRouter API: Not available. Please check your API key.")

    # Predefined queries
    predefined_queries = {
        "Select a predefined query": "",
        "Count all students": "SELECT COUNT(*) as total_students FROM Students;",
        "List all students with details": "SELECT * FROM Students;",
        "Students with CGPA > 8.5": """
            SELECT s.full_name, sr.semester_number, sr.cgpa 
            FROM Students s 
            JOIN SemesterResults sr ON s.student_id = sr.student_id 
            WHERE sr.cgpa > 8.5;
        """,
        "Data Science department students": """
            SELECT s.full_name, ar.department, ar.session_year 
            FROM Students s 
            JOIN AcademicRecords ar ON s.student_id = ar.student_id 
            WHERE ar.department = 'Data Science';
        """,
        "Students with guardian information": """
            SELECT s.full_name, s.roll_number, g.guardian_name, g.relationship, g.guardian_phone
            FROM Students s
            JOIN Guardians g ON s.student_id = g.student_id;
        """
    }

    # Create a dropdown in the top-right corner using columns
    col1, col2 = st.columns([3, 1])
    with col2:
        query_mode = st.selectbox(
            "Query Options",
            ["Ask a Question", "Predefined Query", "Direct SQL"],
            label_visibility="collapsed"
        )

    # Main content area
    st.subheader("Ask a Question About Your Data" if query_mode == "Ask a Question" else 
                "Select a Predefined Query" if query_mode == "Predefined Query" else 
                "Enter a Direct SQL Query")

    # Initialize variables
    question = ""
    custom_sql = ""
    selected_query = "Select a predefined query"

    if query_mode == "Ask a Question":
        if OPENROUTER_AVAILABLE:
            question = st.text_input(
                "Enter your question:",
                placeholder="e.g., Show me all students in the Data Science department",
                label_visibility="collapsed"
            )
        else:
            st.info("AI features are not available. Please switch to Direct SQL mode.")
            
    elif query_mode == "Predefined Query":
        selected_query = st.selectbox("Choose a query:", list(predefined_queries.keys()), label_visibility="collapsed")
        if selected_query != "Select a predefined query":
            st.code(predefined_queries[selected_query], language="sql")
            
    elif query_mode == "Direct SQL":
        custom_sql = st.text_area(
            "Enter your SQL query:",
            height=100,
            placeholder="SELECT * FROM Students;",
            label_visibility="collapsed"
        )

    # Submit button
    submit = st.button("Execute Query", type="primary", use_container_width=True)

    # If submit is clicked
    if submit:
        if query_mode == "Ask a Question" and question and OPENROUTER_AVAILABLE:
            with st.spinner("Processing your question..."):
                response = get_ai_response(question, prompt)
                
                if response:
                    # Clean the SQL response
                    cleaned_sql = clean_sql_query(response)
                    
                    st.subheader("Generated SQL")
                    st.code(cleaned_sql, language="sql")
                    
                    # Execute the query
                    results, columns = read_sql_query(cleaned_sql, 'D:/DESKTOP/VisionID/students_data/mydatabase1.db')
                    
                    if results is not None:
                        display_results(results, columns)
                else:
                    st.error("Could not generate a SQL query. Please try a different question.")
        
        elif query_mode == "Direct SQL" and custom_sql:
            with st.spinner("Executing query..."):
                # Clean the custom SQL input
                cleaned_sql = clean_sql_query(custom_sql)
                results, columns = read_sql_query(cleaned_sql, 'D:/DESKTOP/VisionID/students_data/mydatabase1.db')
                
                if results is not None:
                    display_results(results, columns)
        
        elif query_mode == "Predefined Query" and selected_query != "Select a predefined query":
            with st.spinner("Executing query..."):
                results, columns = read_sql_query(predefined_queries[selected_query], 'D:/DESKTOP/VisionID/students_data/mydatabase1.db')
                
                if results is not None:
                    display_results(results, columns)
        else:
            st.warning("Please provide a valid query.")

    # Database info section
    st.divider()
    expander = st.expander("Database Schema Information")
    with expander:
        st.write("""
        **Tables and Columns:**
        
        - **Students**: student_id, roll_number, full_name, dob, gender, blood_group, nationality, address, phone_number, email
        - **Guardians**: guardian_id, student_id, guardian_name, relationship, guardian_phone, emergency_contact
        - **AcademicRecords**: record_id, student_id, department, session_year, last_cgpa
        - **SemesterResults**: result_id, student_id, semester_number, cgpa, attendance_percentage
        - **FaceEncodes**: encode_id, student_id, image_path, face_encoding, created_at
        - **Faculty**: faculty_id, name, username, password_hash
        """)

    # Add some tips for users
    st.info("💡 **Tip**: For best results with natural language queries, be specific about the columns and tables you're interested in.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("VisionID : Instant Visual Search & Natural Language Analytics for Faculty")