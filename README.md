# NLP Course Recommendation System

This project is a Course Recommendation and Learning Analytics Bot. It uses Natural Language Processing (NLP) to recommend courses based on user queries, PDF uploads, or voice input.

## Features
- **Voice Recommendation**: Speak your interests and get course suggestions.
- **PDF-Based Recommendation**: Upload a PDF (e.g., a syllabus or resume) to get matching courses.
- **Text-Based Search**: Classic search functionality.
- **Learning Analytics**: Visualizes learner progress, skill growth, and feedback distribution.
- **Course Metadata**: Displays university, difficulty level, ratings, and course URLs.

## Dataset
The system uses a Coursera dataset (`data/Coursera.csv`) containing course descriptions, skills, and ratings.

## Tech Stack
- **Python**
- **Streamlit** (UI)
- **Scikit-learn** (TF-IDF, Cosine Similarity)
- **NLTK** (Text Preprocessing)
- **Transformers** (BART for PDF summarization)
- **Matplotlib** (Analytics charts)
- **SpeechRecognition** & **Pyttsx3** (Voice features)

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```
