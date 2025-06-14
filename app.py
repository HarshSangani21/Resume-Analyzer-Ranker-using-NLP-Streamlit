import streamlit as st
import PyPDF2
import docx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def get_similarity_score(jd_text, resume_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)

# Streamlit UI
st.title("📄 Resume Analyzer and Ranker")
st.markdown("Upload a job description and one or more resumes to analyze their fit.")

jd_file = st.file_uploader("Upload Job Description (PDF or DOCX)", type=["pdf", "docx"])
resumes = st.file_uploader("Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)

if jd_file and resumes:
    jd_text = extract_text_from_pdf(jd_file) if jd_file.name.endswith("pdf") else extract_text_from_docx(jd_file)
    jd_text_clean = clean_text(jd_text)

    st.subheader("📊 Resume Fit Scores")
    scores = []

    for resume in resumes:
        res_text = extract_text_from_pdf(resume) if resume.name.endswith("pdf") else extract_text_from_docx(resume)
        res_text_clean = clean_text(res_text)
        score = get_similarity_score(jd_text_clean, res_text_clean)
        scores.append((resume.name, score))
        st.write(f"**{resume.name}** ➤ Fit Score: `{score}%`")

    best = max(scores, key=lambda x: x[1])
    st.success(f"🎯 Best Fit: {best[0]} with a score of {best[1]}%")
