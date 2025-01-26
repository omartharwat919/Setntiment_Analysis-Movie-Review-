import streamlit as st
import tensorflow as tf
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK data (punkt and stopwords)
nltk.download("punkt")
nltk.download("stopwords")

# Load the trained model
try:
    model = tf.keras.models.load_model("sentiment_model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Preprocessing function
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):
    """
    Cleans and preprocesses the input text.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    # Remove <br> tags
    text = re.sub(r"</br>", " ", text).strip()
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    text = " ".join(stemmer.stem(word) for word in tokens if word not in stop_words)
    return text

# Recreate the tokenizer
max_words = 10000  # Must match the value used during training
tokenizer = Tokenizer(num_words=max_words)

# Streamlit app
st.title("IMDb Sentiment Analysis")
st.write("Enter a movie review to analyze its sentiment.")

# Input text box
review = st.text_area("Enter your review here:")

if st.button("Analyze"):
    if review:
        # Clean and preprocess the input
        cleaned_review = clean_text(review)
        # Fit the tokenizer on the input text (this is a hack to make it work)
        tokenizer.fit_on_texts([cleaned_review])
        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([cleaned_review])
        # Pad the sequence
        padded_sequence = pad_sequences(sequence, maxlen=100)  # Must match the value used during training
        # Predict sentiment
        prediction = model.predict(padded_sequence)
        sentiment = "Positive" if prediction[0] > 0.5 else "Negative"
        # Display the result
        st.write(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.write("Please enter a review.")