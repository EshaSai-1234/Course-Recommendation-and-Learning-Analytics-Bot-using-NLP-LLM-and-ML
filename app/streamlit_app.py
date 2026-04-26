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

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

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
    # Base dummy data with all columns
    courses = pd.DataFrame({
        "course_id": [1, 2, 3, 4],
        "title": ["Python for Beginners", "Machine Learning Fundamentals", "Deep Learning with NLP", "Data Science Bootcamp"],
        "skills": ["python basics programming", "machine learning ai models", "nlp bert transformers deep learning", "data science python statistics"],
        "university": ["Self-Paced", "AI University", "DeepMind Academy", "DataCamp"],
        "difficulty_level": ["Beginner", "Intermediate", "Advanced", "Intermediate"],
        "rating": [4.5, 4.6, 4.7, 4.4],
        "course_url": ["#", "#", "#", "#"],
        "description": ["Learn python from scratch.", "Fundamental ML concepts.", "NLP with Transformers.", "Comprehensive data science course."]
    })
    
    coursera_path = "data/Coursera.csv"
    if os.path.exists(coursera_path):
        coursera_df = pd.read_csv(coursera_path)
        # Load all requested columns
        selected_cols = ['Course Name', 'University', 'Difficulty Level', 'Course Rating', 'Course URL', 'Course Description', 'Skills']
        coursera_courses = coursera_df[selected_cols].copy()
        
        coursera_courses.rename(columns={
            'Course Name': 'title', 
            'University': 'university',
            'Difficulty Level': 'difficulty_level', 
            'Course Rating': 'rating',
            'Course URL': 'course_url',
            'Course Description': 'description',
            'Skills': 'skills'
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

def recommend_courses(query, top_n=5, min_rating=0.0):
    query_vec = vectorizer.transform([clean_text(query)])
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    top_indices = similarity[0].argsort()[::-1]
    
    # Filter by rating and limit by top_n
    all_results = courses_df.iloc[top_indices].copy()
    all_results['rating'] = pd.to_numeric(all_results['rating'], errors='coerce')
    filtered_results = all_results[all_results['rating'] >= min_rating]
    
    return filtered_results.head(top_n)

# --- Streamlit UI ---
st.set_page_config(page_title="Course Recommendation Bot", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .main-title { color: #2E5BFF; font-size: 32px; font-weight: bold; text-align: center; width: 100%; }
    .info-box { background-color: #D1D5FF; padding: 15px; border-radius: 5px; color: #333; margin-bottom: 20px; }
    .input-label { background-color: #E2E4FF; padding: 5px 10px; border-radius: 3px; font-weight: bold; display: inline-block; margin-top: 10px; }
    .rec-header { color: #C8466D; font-size: 20px; font-weight: bold; margin-top: 30px; }
    .settings-header { color: #4A4A4A; font-size: 18px; font-weight: bold; margin-top: 20px; }
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
    
    st.markdown('<div class="settings-header">Recommendation Settings</div>', unsafe_allow_html=True)
    top_n = st.number_input("Number of top recommended courses:", min_value=1, max_value=50, value=5)
    min_rating = st.slider("Minimum Rating:", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
    
    st.divider()
    show_analytics = st.toggle("Show Learning Analytics", value=True)
    
    user_query = ""
    if choice == 1:
        st.markdown('<div class="input-label">Voice Input selected</div>', unsafe_allow_html=True)
        mic_on = st.toggle("Microphone", value=True, key="mic_toggle")
        
        if mic_on:
            if 'is_recording' not in st.session_state:
                st.session_state.is_recording = False
            
            rec_col1, rec_col2 = st.columns(2)
            with rec_col1:
                if st.button("🎙️ Start Recording", disabled=st.session_state.is_recording):
                    st.session_state.is_recording = True
                    st.rerun()
            
            with rec_col2:
                if st.button("🛑 Stop Recording", disabled=not st.session_state.is_recording):
                    st.session_state.is_recording = False
                    # Processing happens after rerun
            
            if st.session_state.is_recording:
                st.info("Recording... Please speak now.")
                if sr:
                    recognizer = sr.Recognizer()
                    try:
                        with sr.Microphone() as source:
                            # Adjust for ambient noise
                            recognizer.adjust_for_ambient_noise(source, duration=0.5)
                            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
                        
                        st.session_state.is_recording = False # Auto stop
                        user_query = recognizer.recognize_google(audio)
                        st.success(f"You said: {user_query}")
                    except Exception as e:
                        st.error(f"Voice failed: {e}")
                        st.session_state.is_recording = False
                else:
                    st.warning("Speech recognition not available.")
                    st.session_state.is_recording = False
        else:
            st.info("Microphone is OFF. Toggle it ON to speak.")
        
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
results = None
if user_query:
    st.markdown('<div class="rec-header">⭐ Recommended Courses:</div>', unsafe_allow_html=True)
    results = recommend_courses(user_query, top_n=top_n, min_rating=min_rating)
    
    if not results.empty:
        # Voice Output Option Button
        if pyttsx3:
            if st.button("🔊 Voice Output (Read Results)"):
                with st.spinner("Reading recommendations..."):
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    text_to_read = f"I found {len(results)} courses for you. "
                    for idx, row in results.iterrows():
                        text_to_read += f"Course {idx+1}: {row['title']} by {row['university']}. "
                    
                    engine.say(text_to_read)
                    engine.runAndWait()
                    st.success("Finished reading.")
        
        # Display rich data
        cols_to_show = ['title', 'university', 'difficulty_level', 'rating', 'course_url']
        for idx, row in results.iterrows():
            with st.expander(f"{row['title']} - {row['university']} ({row['rating']} ⭐)"):
                st.write(f"**Difficulty:** {row['difficulty_level']}")
                st.write(f"**Skills:** {row['skills']}")
                st.write(f"**Description:** {row['description']}")
                st.write(f"[Link to Course]({row['course_url']})")
                
        # Also show as dataframe for overview
        st.markdown("### Quick Overview")
        st.dataframe(results[cols_to_show], use_container_width=True)
    else:
        st.warning("No courses found matching your rating criteria.")

# Analytics Section
if show_analytics:
    st.divider()
    st.subheader("Learning Analytics")

    # Dynamic Learner Data generation based on results
    def get_dynamic_learners(results_df):
        # Base data
        base_ids = [101, 102, 103, 104]
        
        if results_df is not None and not results_df.empty:
            # Shift progress/growth based on the average rating of recommendations
            avg_rating = results_df['rating'].mean()
            rating_factor = (avg_rating / 5.0) * 1.2 # Boost progress if high rating results found
            
            progress = [min(100, int(60 * rating_factor)), 
                        min(100, int(75 * rating_factor)), 
                        min(100, int(85 * rating_factor)), 
                        min(100, int(90 * rating_factor))]
            
            growth = [min(100, int(50 * rating_factor)), 
                      min(100, int(70 * rating_factor)), 
                      min(100, int(80 * rating_factor)), 
                      min(100, int(95 * rating_factor))]
            
            # Simulated feedback tied to difficulty
            if 'difficulty_level' in results_df.columns:
                most_common_diff = results_df['difficulty_level'].mode().iloc[0]
                if most_common_diff == 'Beginner':
                    feedback = ["Excellent", "Very Good", "Excellent", "Good"]
                else:
                    feedback = ["Good", "Good", "Very Good", "Excellent"]
            else:
                feedback = ["Good", "Very Good", "Excellent", "Excellent"]
        else:
            # Default static data
            progress = [60, 75, 85, 90]
            growth = [50, 70, 80, 95]
            feedback = ["Good", "Very Good", "Excellent", "Excellent"]
            
        return pd.DataFrame({
            "learner_id": base_ids,
            "progress": progress,
            "skill_growth": growth,
            "feedback": feedback
        })

    learners = get_dynamic_learners(results)

    # Rows for charts
    row1_col1, row1_col2, row1_col3 = st.columns(3)

    with row1_col1:
        fig1, ax1 = plt.subplots()
        ax1.bar(learners["learner_id"].astype(str), learners["progress"], color='#2E5BFF')
        ax1.set_title("Learner Progress")
        ax1.set_ylabel("Progress %")
        ax1.set_xlabel("Learner ID")
        st.pyplot(fig1)

    with row1_col2:
        fig2, ax2 = plt.subplots()
        ax2.bar(learners["learner_id"].astype(str), learners["skill_growth"], color='#C8466D')
        ax2.set_title("Skill Development Growth")
        ax2.set_ylabel("Skill Growth %")
        ax2.set_xlabel("Learner ID")
        st.pyplot(fig2)

    with row1_col3:
        feedback_counts = learners["feedback"].value_counts()
        fig3, ax3 = plt.subplots()
        ax3.pie(feedback_counts, labels=feedback_counts.index, autopct='%1.1f%%', startangle=90, colors=['#D1D5FF', '#E2E4FF', '#2E5BFF', '#C8466D'])
        ax3.set_title("Learner Feedback Distribution")
        st.pyplot(fig3)

    # Secondary analytics based on recommendations
    if results is not None and not results.empty:
        st.markdown("### Recommendation Insights")
        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1:
            if 'university' in results.columns:
                univ_counts = results['university'].value_counts().head(10)
                fig4, ax4 = plt.subplots()
                univ_counts.plot(kind='bar', ax=ax4, color='orange')
                ax4.set_title("Top Universities (Recommended)")
                ax4.set_ylabel("Count")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig4)

        with row2_col2:
            if 'difficulty_level' in results.columns:
                diff_counts = results['difficulty_level'].value_counts()
                fig5, ax5 = plt.subplots()
                ax5.pie(diff_counts, labels=diff_counts.index, autopct='%1.1f%%', startangle=90)
                ax5.set_title("Difficulty Distribution")
                st.pyplot(fig5)

        row3_col1, row3_col2 = st.columns(2)
        
        with row3_col1:
            if 'skills' in results.columns:
                all_skills = results['skills'].dropna().str.split(',', expand=True).stack().str.strip()
                top_skills = all_skills.value_counts().head(10)
                fig6, ax6 = plt.subplots()
                ax6.pie(top_skills, labels=top_skills.index, autopct='%1.1f%%', startangle=90)
                ax6.set_title("Top Skills in Recommendations")
                st.pyplot(fig6)

        with row3_col2:
            if 'difficulty_level' in results.columns and 'rating' in results.columns:
                avg_rating_diff = results.groupby('difficulty_level')['rating'].mean()
                fig7, ax7 = plt.subplots()
                avg_rating_diff.plot(kind='bar', ax=ax7, color='purple')
                ax7.set_title("Avg Rating by Difficulty")
                ax7.set_ylabel("Rating")
                st.pyplot(fig7)

        row4_col1, row4_col2 = st.columns(2)
        with row4_col1:
            if 'rating' in results.columns:
                fig8, ax8 = plt.subplots()
                results['rating'].plot(kind='hist', bins=10, ax=ax8, color='cyan', edgecolor='black')
                ax8.set_title("Course Ratings Distribution")
                ax8.set_xlabel("Rating")
                st.pyplot(fig8)
        
        with row4_col2:
             st.info("The charts above provide a comprehensive breakdown of the recommended courses based on University, Difficulty, Skills, and Ratings.")
