import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# بارگذاری مدل
model = joblib.load("models/sentiment_model.pkl")

# تابع پاک‌سازی متن
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# رابط کاربری با Streamlit
st.title("🧠 Sentiment Analyzer")
st.write("این یک ابزار ساده برای تشخیص احساسات نظرات متنی است.")

user_input = st.text_area("متن نظر خود را وارد کنید:")

if st.button("تحلیل احساس"):
    cleaned = clean_text(user_input)
    prediction = model.predict([cleaned])[0]
    label = "مثبت 😊" if prediction == 1 else "منفی 😞"
    st.success(f"نتیجه تحلیل: **{label}**")
