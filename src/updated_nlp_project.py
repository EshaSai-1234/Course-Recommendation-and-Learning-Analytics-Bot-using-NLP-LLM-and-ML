# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import sys

# Optional imports for robustness
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words("english"))
except ImportError:
    print("Warning: NLTK not found. Text processing may be limited.")
    stop_words = set()

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Error: scikit-learn not found. Install it to run recommendations.")
    sys.exit(1)

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

# ---------------------------------------------------------
# Global Setup
# ---------------------------------------------------------

# Voice Engine Setup
engine = None
if pyttsx3:
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)
    except Exception as e:
        print(f"Warning: Could not initialize TTS engine: {e}")

# Summarizer (Lazy Loading)
summarizer_pipeline = None

def get_summarizer():
    global summarizer_pipeline
    if summarizer_pipeline is None and pipeline:
        print("Loading summarization model... (this may take a moment)")
        summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer_pipeline

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

def speak(text):
    if engine:
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    try:
        print(f"[Bot]: {text}")
    except UnicodeEncodeError:
        try:
            print(f"[Bot]: {text.encode('ascii', 'replace').decode()}")
        except:
            pass

def voice_to_text():
    if not sr:
        print("Speech recognition not available.")
        return ""
    
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("[Mic] Speak now...")
            audio = recognizer.listen(source, timeout=5)
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except Exception as e:
        print("Voice recognition failed. Please type your input.")
        return ""

def extract_pdf_text(pdf_path):
    if not PdfReader:
        print("pypdf not installed.")
        return ""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def summarize_text(text):
    summ_pipe = get_summarizer()
    if not summ_pipe:
        return text[:500] + "..."
    
    # Chunk text to avoid max length errors
    text = text[:3000] 
    try:
        summary = summ_pipe(text, max_length=130, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Summarization error: {e}")
        return text[:500]

# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------

print("Initializing Course Database...")

# Default Courses
courses = pd.DataFrame({
    "course_id": [1,2,3,4],
    "title": [
        "Python for Beginners",
        "Machine Learning Fundamentals",
        "Deep Learning with NLP",
        "Data Science Bootcamp"
    ],
    "skills": [
        "python basics programming",
        "machine learning ai models",
        "nlp bert transformers deep learning",
        "data science python statistics"
    ],
    "rating": [4.5, 4.6, 4.7, 4.4]
})

coursera_path = "data/Coursera.csv"
if os.path.exists(coursera_path):
    print(f"Loading external data from {coursera_path}...")
    try:
        coursera_df = pd.read_csv(coursera_path)
        
        coursera_courses = coursera_df[['Course Name', 'Skills', 'University', 'Difficulty Level', 'Course Rating']].copy()
        coursera_courses.rename(columns={
            'Course Name': 'title',
            'Skills': 'skills',
            'University': 'university',
            'Difficulty Level': 'difficulty_level',
            'Course Rating': 'rating'
        }, inplace=True)
        
        # Merge
        coursera_courses['course_id'] = np.arange(len(courses), len(courses) + len(coursera_courses))
        courses = pd.concat([courses, coursera_courses], ignore_index=True)
        print(f"Total courses loaded: {len(courses)}")
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
else:
    print(f"Warning: {coursera_path} not found. Using default courses only.")

# Preprocessing
courses["clean_skills"] = courses["skills"].apply(clean_text)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(courses["clean_skills"])

def recommend_courses(query, top_n=5):
    query_vec = vectorizer.transform([clean_text(query)])
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    top_indices = similarity[0].argsort()[-top_n:][::-1]
    return courses.iloc[top_indices]

# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------

def save_plots(results):
    prefix = "updated_project_output"
    
    # 1. University
    if 'university' in results.columns:
        counts = results['university'].value_counts().head(10)
        if not counts.empty:
            plt.figure(figsize=(10,5))
            plt.bar(counts.index, counts.values, color='skyblue')
            plt.title("Top Universities (Recommended)")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{prefix}_universities.png")
            plt.close()
    
    # 2. Rating by Difficulty
    if 'difficulty_level' in results.columns and 'rating' in results.columns:
        results['rating_num'] = pd.to_numeric(results['rating'], errors='coerce')
        avg = results.groupby('difficulty_level')['rating_num'].mean()
        if not avg.empty:
            plt.figure(figsize=(6,5))
            plt.bar(avg.index, avg.values, color='orange')
            plt.title("Avg Rating by Difficulty")
            plt.tight_layout()
            plt.savefig(f"{prefix}_rating_diff.png")
            plt.close()
            
    # 3. Difficulty Pie
    if 'difficulty_level' in results.columns:
        counts = results['difficulty_level'].value_counts()
        if not counts.empty:
            plt.figure(figsize=(6,6))
            plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
            plt.title("Difficulty Levels")
            plt.savefig(f"{prefix}_difficulty.png")
            plt.close()

    print("Plots saved to updated_project_output_*.png")

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def main():
    print("\n--- NLP Recommendation System (Updated) ---")
    print("1. Voice Input")
    print("2. PDF Upload")
    print("3. Text Input")
    
    choice = input("Enter choice: ").strip()
    
    query = ""
    if choice == "1":
        query = voice_to_text()
        if not query:
            query = input("Voice failed. Enter your query: ").strip()
    elif choice == "2":
        path = input("Enter PDF path: ").strip()
        if os.path.exists(path):
            text = extract_pdf_text(path)
            query = summarize_text(text)
            print(f"Summary: {query}")
        else:
            print("File not found.")
            return
    elif choice == "3":
        query = input("Enter your query: ").strip()
    else:
        print("Invalid choice.")
        return

    if not query:
        print("No query provided. Exiting.")
        return

    print(f"\nSearching for: {query}")
    results = recommend_courses(query, top_n=10)
    
    # Output Results
    print("\nTop Recommendations:")
    columns = ['title', 'rating', 'university', 'difficulty_level']
    cols_to_show = [c for c in columns if c in results.columns]
    
    try:
        print(results[cols_to_show].to_string(index=False))
    except UnicodeEncodeError:
        print("Results contains characters that cannot be printed to this console.")
        print("Saving to file...")
    
    # Save to file
    with open("updated_project_results.txt", "w", encoding="utf-8") as f:
        f.write(results.to_string(index=False))
    print("\nFull results saved to updated_project_results.txt")
    
    # Speak top 3
    for title in results["title"].head(3):
        speak(f"Recommend: {title}")
        
    # Generate Plots
    save_plots(results)

if __name__ == "__main__":
    main()
