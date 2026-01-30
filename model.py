from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

texts = [
    "I have experience in python and data science",
    "Worked on machine learning projects",
    "Strong analytical and problem solving skills",
    "I am an AI language model trained by OpenAI",
    "This resume was generated using artificial intelligence"
]

labels = [0, 0, 0, 1, 1]  # 0 = Human, 1 = AI

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

joblib.dump(model, "ai_detector.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("AI model trained successfully")
