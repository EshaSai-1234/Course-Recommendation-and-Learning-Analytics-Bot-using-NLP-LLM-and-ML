import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import sys
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional imports with fallbacks
try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

# --- Setup NLTK ---
@st.cache_resource
def setup_nltk():
    nltk.download('stopwords')
    return set(stopwords.words("english"))

stop_words = setup_nltk()

# --- Utility Functions ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

@st.cache_resource
def get_summarizer():
    if pipeline:
        try:
            return pipeline("summarization", model="facebook/bart-large-cnn")
        except:
            return None
    return None

def summarize_text(text):
    summarizer = get_summarizer()
    if summarizer:
        text = text[:3000]
        try:
            summary = summarizer(text, max_length=130, min_length=50, do_sample=False)
            return summary[0]['summary_text']
        except:
            pass
    # Fallback
    return text[:200] + "..."

def extract_pdf_text(pdf_file):
    if not PdfReader:
        return "pypdf not installed."
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

# --- Data Loading ---
@st.cache_data
def load_data():
    courses = pd.DataFrame({
        "course_id": [1,2,3,4],
        "title": ["Python for Beginners", "Machine Learning Fundamentals", "Deep Learning with NLP", "Data Science Bootcamp"],
        "skills": ["python basics programming", "machine learning ai models", "nlp bert transformers deep learning", "data science python statistics"],
        "rating": [4.5, 4.6, 4.7, 4.4]
    })
    
    coursera_path = "data/Coursera.csv"
    if os.path.exists(coursera_path):
        coursera_df = pd.read_csv(coursera_path)
        coursera_courses = coursera_df[['Course Name', 'Skills', 'University', 'Difficulty Level', 'Course Rating']].copy()
        coursera_courses.rename(columns={
            'Course Name': 'title', 'Skills': 'skills', 'University': 'university',
            'Difficulty Level': 'difficulty_level', 'Course Rating': 'rating'
        }, inplace=True)
        coursera_courses['course_id'] = np.arange(len(courses), len(courses) + len(coursera_courses))
        courses = pd.concat([courses, coursera_courses], ignore_index=True)
    
    courses["clean_skills"] = courses["skills"].apply(clean_text)
    return courses

courses_df = load_data()

# --- Recommendation Logic ---
@st.cache_resource
def get_vectorizer(data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data["clean_skills"])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = get_vectorizer(courses_df)

def recommend_courses(query, top_n=5):
    query_vec = vectorizer.transform([clean_text(query)])
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    top_indices = similarity[0].argsort()[-top_n:][::-1]
    return courses_df.iloc[top_indices]

# --- Streamlit UI ---
st.set_page_config(page_title="Course Recommendation Bot", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .main-title { color: #2E5BFF; font-size: 32px; font-weight: bold; }
    .info-box { background-color: #D1D5FF; padding: 15px; border-radius: 5px; color: #333; margin-bottom: 20px; }
    .input-label { background-color: #E2E4FF; padding: 5px 10px; border-radius: 3px; font-weight: bold; display: inline-block; margin-top: 10px; }
    .rec-header { color: #C8466D; font-size: 20px; font-weight: bold; margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Course Recommendation and Learning Analytics Bot</div>', unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("""
    <div class="info-box">
        Choose input type:<br>
        1. Voice Input<br>
        2. PDF Upload<br>
        3. Text Input
    </div>
    """, unsafe_allow_html=True)
    
    choice = st.selectbox("Enter choice:", [1, 2, 3], index=2)
    
    user_query = ""
    if choice == 1:
        st.markdown('<div class="input-label">Voice Input selected</div>', unsafe_allow_html=True)
        if st.button("🎙️ Start Recording"):
            if sr:
                recognizer = sr.Recognizer()
                try:
                    with sr.Microphone() as source:
                        st.write("Listening...")
                        audio = recognizer.listen(source, timeout=5)
                    user_query = recognizer.recognize_google(audio)
                    st.success(f"You said: {user_query}")
                except Exception as e:
                    st.error(f"Voice failed: {e}")
            else:
                st.warning("Speech recognition not available. Please use Text Input.")
        
    elif choice == 2:
        st.markdown('<div class="input-label">PDF Upload selected</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file:
            with st.spinner("Extracting and summarizing..."):
                pdf_text = extract_pdf_text(uploaded_file)
                user_query = summarize_text(pdf_text)
                st.info(f"Summary: {user_query}")
                
    elif choice == 3:
        st.markdown('<div class="input-label">Text Input selected</div>', unsafe_allow_html=True)
        user_query = st.text_input("Enter your query:", placeholder="e.g. Machine Learning")

# Display Recommendations
if user_query:
    st.markdown('<div class="rec-header">⭐ Recommended Courses:</div>', unsafe_allow_html=True)
    results = recommend_courses(user_query, top_n=5)
    
    # Clean up display columns
    cols_to_show = ['course_id', 'title', 'skills', 'rating']
    if 'clean_skills' in results.columns:
        cols_to_show.append('clean_skills')
    
    st.dataframe(results[cols_to_show], use_container_width=True)

# Analytics Section
st.divider()
st.subheader("Learning Analytics")

learners = pd.DataFrame({
    "learner_id": [101, 102, 103, 104],
    "progress": [60, 75, 85, 90],
    "skill_growth": [50, 70, 80, 95],
    "feedback": ["Good", "Very Good", "Excellent", "Excellent"]
})

# Graphs
chart_col1, chart_col2, chart_col3 = st.columns(3)

with chart_col1:
    fig1, ax1 = plt.subplots()
    ax1.bar(learners["learner_id"].astype(str), learners["progress"])
    ax1.set_title("Learner Progress")
    ax1.set_ylabel("Progress %")
    ax1.set_xlabel("Learner ID")
    st.pyplot(fig1)

with chart_col2:
    fig2, ax2 = plt.subplots()
    ax2.bar(learners["learner_id"].astype(str), learners["skill_growth"])
    ax2.set_title("Skill Development Growth")
    ax2.set_ylabel("Skill Growth %")
    ax2.set_xlabel("Learner ID")
    st.pyplot(fig2)

with chart_col3:
    feedback_counts = learners["feedback"].value_counts()
    fig3, ax3 = plt.subplots()
    ax3.pie(feedback_counts, labels=feedback_counts.index, autopct='%1.1f%%', startangle=90)
    ax3.set_title("Learner Feedback Distribution")
    st.pyplot(fig3)
