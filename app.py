import streamlit as st
import spacy
import gdown
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from textblob import TextBlob
from pathlib import Path
import os

# Function to load spaCy model from a local directory
def load_spacy_model(model_path):
    try:
        nlp = spacy.load(model_path)
        return nlp
    except OSError as e:
        st.error(f"Error loading spaCy model: {e}")

# Function to download model from Google Drive using gdown
def download_model_from_drive():
    # Define the Google Drive file ID of your model folder
    file_id = 'your_file_id_here'

    # Automatically assign a path for saving the model folder
    output_path = Path("spaCy_model")

    # Check if the model folder already exists, if not, download it
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        output_file = f'{output_path}.zip'
        gdown.download(url, output_file, quiet=False)
        st.write("Extracting model folder...")
        os.system(f"unzip -q {output_file} -d {output_path}")
        os.remove(output_file)

    return output_path

# Function to get the transcript from a YouTube video URL
def get_transcript(youtube_url):
    try:
        video_id = YouTube(youtube_url).video_id
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ' '.join([entry['text'] for entry in transcript])
        return text
    except Exception as e:
        return f"Error: {e}"

# Function to analyze sentiment of a sentence
def analyze_sentiment(sentence):
    blob = TextBlob(sentence)
    return blob.sentiment.polarity

# Function for advanced NLP analysis
def advanced_nlp_analysis(text, nlp):
    doc = nlp(text)
    entities = list(set((ent.text, ent.label_) for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE']))
    key_phrases = [chunk.text for chunk in doc.noun_chunks]
    sentiment_score = TextBlob(text).sentiment.polarity
    top_entities = sorted(entities, key=lambda x: text.count(x[0]), reverse=True)[:10]
    top_key_phrases = key_phrases[:10]
    sentences_with_sentiments = [(sent.text, analyze_sentiment(sent.text)) for sent in doc.sents]
    top_emotional_sentences = sorted(sentences_with_sentiments, key=lambda x: abs(x[1]), reverse=True)[:10]
    top_negative_sentences = [sent for sent in top_emotional_sentences if sent[1] < 0][:5]
    top_positive_sentences = [sent for sent in top_emotional_sentences if sent[1] > 0][:5]
    summary_text = ' '.join([sent.text for sent in doc.sents][:5])
    return top_entities, top_key_phrases, sentiment_score, top_negative_sentences, top_positive_sentences, summary_text

def print_summary(title, values):
    summary = f"{title}:\n"
    for idx, value in enumerate(values, start=1):
        if isinstance(value, list):
            for sub_idx, sub_value in enumerate(value, start=1):
                summary += f"  {idx}.{sub_idx}. {sub_value[0]} (Sentiment: {sub_value[1]:.2f})\n"
        else:
            summary += f"  {idx}. {value}\n"
    return summary

def main():
    st.title("YouTube Video NLP Insights")

    # Load spaCy model
    model_path = download_model_from_drive()
    nlp = load_spacy_model(model_path)

    # UI for input URL
    youtube_url = st.text_input("Enter YouTube Video URL")
    if st.button("Analyze"):
        st.write("Analyzing...")

        # Get video text from URL
        video_text = get_transcript(youtube_url)

        if video_text.startswith("Error"):
            st.error(video_text)
        else:
            # Perform advanced NLP analysis
            top_entities, top_key_phrases, sentiment_score, top_negative_sentences, top_positive_sentences, summary_text = advanced_nlp_analysis(video_text, nlp)

            # Display analysis results
            st.subheader("Top 10 Named Entities")
            st.write(print_summary("Top 10 Named Entities", top_entities))

            st.subheader("Top 10 Key Phrases")
            st.write(print_summary("Top 10 Key Phrases", top_key_phrases))

            # Provide a sentiment summary
            sentiment_summary = "Positive" if sentiment_score > 0 else "Neutral" if sentiment_score == 0 else "Negative"
            st.write(f"Sentiment Summary: {sentiment_summary} (Score: {sentiment_score:.2f})")

            st.subheader("Top 5 Negative Sentences")
            st.write(print_summary("Top 5 Negative Sentences", top_negative_sentences))

            st.subheader("Top 5 Positive Sentences")
            st.write(print_summary("Top 5 Positive Sentences", top_positive_sentences))

            st.subheader("Summary of the Video Text (first 100 words)")
            st.write(summary_text)

if __name__ == "__main__":
    main()
