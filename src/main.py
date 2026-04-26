# -*- coding: utf-8 -*-

import nltk
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

from pypdf import PdfReader
from transformers import pipeline

# -------------------------
# NLTK
# -------------------------
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

# -------------------------
# Utility
# -------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z ]', '', text)
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# -------------------------
# Voice
# -------------------------
def voice_to_text():
    if sr is None:
        print("Speech recognition module not found. Switching to text input.")
        return ""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("[Mic] Speak now...")

            audio = recognizer.listen(source, timeout=5)
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except Exception as e:
        print("Voice recognition failed or no microphone found. Switching to text input.")
        return ""

try:
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    if voices:
        engine.setProperty('voice', voices[0].id)
except:
    engine = None

def speak(text):
    if engine:
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    print(f"[Bot]: {text}")

# -------------------------
# PDF
# -------------------------
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t
    return text

# -------------------------
# Summarizer
# -------------------------
summarizer = None

def get_summarizer():
    global summarizer
    if summarizer is None:
        print("Loading summarization model... (first time use may be slow)")
        summarizer = pipeline(task="text2text-generation",
                              model="facebook/bart-large-cnn")
    return summarizer

def summarize_text(text):
    text = text[:3000]
    summ_pipe = get_summarizer()
    out = summ_pipe(text, max_length=130)
    return out[0]["generated_text"]

# -------------------------
# Initial sample courses
# -------------------------
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

courses["clean_skills"] = courses["skills"].apply(clean_text)

# -------------------------
# Load Coursera dataset
# -------------------------
print("Loading Coursera dataset...")

# Put Coursera.csv in the SAME folder as main.py
coursera_df = pd.read_csv("data/Coursera.csv")

coursera_courses = coursera_df[
    ['Course Name','Skills','University','Difficulty Level','Course Rating']
].copy()

coursera_courses.rename(columns={
    'Course Name':'title',
    'Skills':'skills',
    'University':'university',
    'Difficulty Level':'difficulty_level',
    'Course Rating':'rating'
}, inplace=True)

coursera_courses["clean_skills"] = coursera_courses["skills"].apply(clean_text)
coursera_courses["course_id"] = np.arange(len(courses)+1,
                                          len(courses)+1+len(coursera_courses))

courses = pd.concat([
    courses,
    coursera_courses[
        ["course_id","title","skills","rating",
         "clean_skills","university","difficulty_level"]
    ]
], ignore_index=True)

# -------------------------
# Vectorizer
# -------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(courses["clean_skills"])

def recommend_courses(query, top_n=len(courses)):
    query_vec = vectorizer.transform([clean_text(query)])
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    top_indices = similarity[0].argsort()[-top_n:][::-1]
    return courses.iloc[top_indices]

# -------------------------
# Learner visualisation
# -------------------------
learners = pd.DataFrame({
    "learner_id": [101,102,103,104],
    "progress": [60, 75, 85, 90],
    "skill_growth": [50, 70, 80, 95],
    "feedback": ["Good", "Very Good", "Excellent", "Excellent"],
    "rating": [4, 5, 5, 4]
})

plt.figure()
plt.bar(learners["learner_id"], learners["progress"])
plt.xlabel("Learner ID")
plt.ylabel("Progress %")
plt.title("Learner Progress")
plt.savefig("output_learner_progress.png")
plt.close()


plt.figure()
plt.bar(learners["learner_id"], learners["skill_growth"])
plt.xlabel("Learner ID")
plt.ylabel("Skill Growth %")
plt.title("Skill Development Growth")
plt.savefig("output_skill_growth.png")
plt.close()


feedback_counts = learners["feedback"].value_counts()
plt.figure()
plt.pie(feedback_counts, labels=feedback_counts.index, autopct="%1.1f%%")
plt.title("Learner Feedback Distribution")
plt.savefig("output_feedback_dist.png")
plt.close()


rating_counts = learners["rating"].value_counts()
plt.figure()
plt.pie(rating_counts, labels=rating_counts.index, autopct="%1.1f%%")
plt.title("Course Ratings")
plt.savefig("output_rating_dist.png")
plt.close()


# -------------------------
# User input
# -------------------------
print("\nChoose input type:")
print("1. Voice Input")
print("2. PDF Upload")
print("3. Text Input")

choice = input("Enter choice: ")

if choice == "1":
    user_query = voice_to_text()
    results = recommend_courses(user_query)

elif choice == "2":
    pdf_path = input("Enter PDF file path: ")
    pdf_text = extract_pdf_text(pdf_path)
    summary = summarize_text(pdf_text)
    print("\n[PDF] PDF Summary:\n", summary)

    speak(summary)
    results = recommend_courses(summary)

else:
    user_query = input("Enter your query: ")
    results = recommend_courses(user_query)

# -------------------------
# Output
# -------------------------
print("\n--> Recommended Courses:\n")

# Save to file
with open("output_recommendations.txt", "w", encoding="utf-8") as f:
    f.write(results[["title","university","difficulty_level","rating"]].to_string(index=False))

print("Results saved to output_recommendations.txt")

# Safe print to console
try:
    print(results[["title","university","difficulty_level","rating"]].to_string(index=False))
except UnicodeEncodeError:
    print("Could not print full results to console due to encoding issues. See text file.")

for title in results["title"].head(5):
    speak(f"I recommend {title}")

# -------------------------
# Plots on recommendations
# -------------------------
courses_per_univ = results['university'].value_counts()
plt.figure(figsize=(10,5))
plt.bar(courses_per_univ.index, courses_per_univ.values)
plt.xlabel("University")
plt.ylabel("Number of Courses")
plt.title("Recommended Courses per University")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("output_rec_univ.png")
plt.close()


results["rating"] = pd.to_numeric(results["rating"], errors="coerce")
avg_rating = results.groupby('difficulty_level')['rating'].mean()
plt.figure(figsize=(6,5))
plt.bar(avg_rating.index, avg_rating.values)
plt.xlabel("Difficulty Level")
plt.ylabel("Average Rating")
plt.title("Average Rating by Difficulty Level")
plt.tight_layout()
plt.savefig("output_rec_avg_rating.png")
plt.close()


difficulty_counts = results['difficulty_level'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(difficulty_counts,
        labels=difficulty_counts.index,
        autopct='%1.1f%%',
        startangle=90)
plt.title("Recommended Courses by Difficulty Level")
plt.savefig("output_rec_difficulty.png")
plt.close()


all_skills = results['skills'].dropna().str.split(',', expand=True)\
                               .stack().str.strip()
top_skills = all_skills.value_counts().head(10)

plt.figure(figsize=(7,7))
plt.pie(top_skills,
        labels=top_skills.index,
        autopct='%1.1f%%',
        startangle=90)
plt.title("Top Skills in Recommended Courses")
plt.savefig("output_rec_top_skills.png")
plt.close()


# -------------------------
# Text output
# -------------------------
print("\nRecommended Courses (Full Table):\n")
# Save full table
with open("output_full_recommendations.txt", "w", encoding="utf-8") as f:
    f.write(results.to_string(index=False))

try:
    print(results.to_string(index=False))
except UnicodeEncodeError:
    print("Could not print full table to console. Saved to output_full_recommendations.txt")

