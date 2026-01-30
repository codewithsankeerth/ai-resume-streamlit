import streamlit as st
import PyPDF2
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("ai_detector.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}
.title {
    font-size: 40px;
    font-weight: 700;
}
.subtitle {
    color: #6b7280;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🤖 AI Resume Screening System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart • Fast • AI-Powered Resume Analysis</div>', unsafe_allow_html=True)
st.write("")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📌 Job Description")
    job_description = st.text_area(
        "Paste job description here",
        "Python developer with machine learning and data science skills"
    )
    st.markdown("---")
    st.info("Upload a resume to evaluate match score and AI content detection.")

# ---------------- FILE UPLOAD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📄 Upload Resume (PDF only)", type=["pdf"])
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TEXT EXTRACTION ----------------
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# ---------------- ANALYSIS ----------------
if uploaded_file:
    with st.spinner("🔍 Analyzing resume..."):
        resume_text = extract_text(uploaded_file)

        X = vectorizer.transform([resume_text])
        ai_pred = model.predict(X)[0]
        ai_result = "AI Generated 🤖" if ai_pred == 1 else "Human Written 👤"

        jd_vec = vectorizer.transform([job_description])
        match_score = cosine_similarity(X, jd_vec)[0][0] * 100

    st.success("✅ Analysis Complete")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Resume Evaluation")

    st.write(f"**Resume Type:** {ai_result}")
    st.write("**Job Match Score:**")

    st.progress(min(int(match_score), 100))
    st.write(f"🔹 {round(match_score, 2)}% match")

    if match_score >= 70:
        st.success("🎯 Strong match for the role")
    elif match_score >= 40:
        st.warning("⚠️ Partial match – needs improvement")
    else:
        st.error("❌ Low match for this role")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with Streamlit • Machine Learning • Python")

    font-weight: 700;
}
.subtitle {
    color: #6b7280;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🤖 AI Resume Screening System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart • Fast • AI-Powered Resume Analysis</div>', unsafe_allow_html=True)
st.write("")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📌 Job Description")
    job_description = st.text_area(
        "Paste job description here",
        "Python developer with machine learning and data science skills"
    )
    st.markdown("---")
    st.info("Upload a resume to evaluate match score and AI content detection.")

# ---------------- FILE UPLOAD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📄 Upload Resume (PDF only)", type=["pdf"])
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- TEXT EXTRACTION ----------------
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# ---------------- ANALYSIS ----------------
if uploaded_file:
    with st.spinner("🔍 Analyzing resume..."):
        resume_text = extract_text(uploaded_file)

        X = vectorizer.transform([resume_text])
        ai_pred = model.predict(X)[0]
        ai_result = "AI Generated 🤖" if ai_pred == 1 else "Human Written 👤"

        jd_vec = vectorizer.transform([job_description])
        match_score = cosine_similarity(X, jd_vec)[0][0] * 100

    st.success("✅ Analysis Complete")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Resume Evaluation")

    st.write(f"**Resume Type:** {ai_result}")
    st.write("**Job Match Score:**")

    st.progress(min(int(match_score), 100))
    st.write(f"🔹 {round(match_score, 2)}% match")

    if match_score >= 70:
        st.success("🎯 Strong match for the role")
    elif match_score >= 40:
        st.warning("⚠️ Partial match – needs improvement")
    else:
        st.error("❌ Low match for this role")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with Streamlit • Machine Learning • Python")
