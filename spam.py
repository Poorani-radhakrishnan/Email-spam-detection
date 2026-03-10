import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("About Project")
st.sidebar.write("""
This app detects whether a message is **Spam or Not Spam** using a Machine Learning model.

Model used:
- Multinomial Naive Bayes

Steps:
1. Text Vectorization
2. Model Prediction
""")

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv("spam.csv", encoding="latin-1")

data = data[['v1','v2']]
data.columns = ['label','message']

data['label'] = data['label'].map({'ham':0,'spam':1})

# -------------------------------
# Train Model
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2
)

vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -------------------------------
# Main UI
# -------------------------------
st.title("📧 Spam Email Classifier")
st.markdown("---")

st.write("Enter a message below to check whether it is **Spam or Not Spam**.")

with st.container():

    message = st.text_area(
        "✉️ Enter your message",
        height=150,
        placeholder="Example: Congratulations! You have won a free prize..."
    )

    col1, col2 = st.columns([1,1])

    with col1:
        check = st.button("🔍 Check Message")

    with col2:
        clear = st.button("🗑 Clear")

# -------------------------------
# Prediction
# -------------------------------
if check:

    if message.strip() == "":
        st.warning("⚠️ Please enter a message first.")

    else:
        msg_vector = vectorizer.transform([message])
        prediction = model.predict(msg_vector)[0]

        st.markdown("### Result")

        if prediction == 1:
            st.error("🚨 This message is **SPAM**")
        else:
            st.success("✅ This message is **NOT Spam**")

st.markdown("---")
