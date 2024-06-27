from textblob import TextBlob
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import cleantext
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re

st.header('Sentiment Analysis')

# Function to clean text
def clean_text(text):
    cleaned_text = cleantext.clean(text, clean_all=False, extra_spaces=True,
                                   stopwords=True, lowercase=True, numbers=True, punct=True)
    return cleaned_text

# Function to create a word cloud
def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Function to analyze sentiment and create bar graph
def analyze_sentiment(text):
    cleaned_text = clean_text(text)
    blob = TextBlob(cleaned_text)
    polarity = round(blob.sentiment.polarity, 2)
    subjectivity = round(blob.sentiment.subjectivity, 2)
    
    # Display cleaned text
    st.write('Cleaned Text:')
    st.write(cleaned_text)
    
    # Display polarity and subjectivity
    st.write('Polarity: ', polarity)
    st.write('Subjectivity: ', subjectivity)
    
    # Create word cloud
    st.subheader('Word Cloud')
    create_wordcloud(cleaned_text)

    # Determine sentiment category
    if polarity > 0:
        sentiment = 'Positive'
    elif polarity < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    # Bar graph
    st.subheader('Sentiment Distribution')
    plt.figure(figsize=(6, 4))
    sns.barplot(x=['Positive', 'Negative', 'Neutral'], y=[polarity > 0, polarity < 0, polarity == 0])
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Distribution')
    st.pyplot(plt)

# UI for text input and analysis
with st.expander('Analyze Text'):
    text = st.text_area('Enter your text here:')
    clean_checkbox = st.checkbox('Clean Text')

    if st.button('Analyze'):
        if clean_checkbox:
            text = clean_text(text)
        analyze_sentiment(text)

