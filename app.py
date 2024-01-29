import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from textblob import TextBlob
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize, sent_tokenize
from nltk.tree import Tree

# Download NLTK resources
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

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

# Function for advanced NLP analysis using NLTK
def advanced_nlp_analysis(text):
    entities = []
    key_phrases = []

    # Tokenize text into sentences using NLTK
    sentences = sent_tokenize(text)

    # Named Entity Recognition using NLTK
    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)
        chunked_words = ne_chunk(tagged_words)

        for subtree in chunked_words:
            if isinstance(subtree, Tree):
                entity = " ".join([word for word, pos in subtree.leaves()])
                entities.append((entity, subtree.label()))
            else:
                key_phrases.append(subtree[0])

    # Sort entities by length in descending order
    entities.sort(key=lambda x: len(x[0]), reverse=True)
    top_entities = entities[:10]

    # Additional sentiment analysis using TextBlob
    sentences_with_sentiments = [(sentence, analyze_sentiment(sentence)) for sentence in sentences]
    top_emotional_sentences = sorted(sentences_with_sentiments, key=lambda x: abs(x[1]), reverse=True)
    top_negative_sentences = [sent for sent in top_emotional_sentences if sent[1] < 0][:5]
    top_positive_sentences = [sent for sent in top_emotional_sentences if sent[1] > 0][:5]

    return top_entities, top_positive_sentences, top_negative_sentences, text

def main():
    st.title("YouTube Video NLP Insights")
    
    # UI for input URL
    youtube_url = st.text_input("Enter YouTube Video URL")
    if st.button("Analyze"):
        st.write("Analyzing...")

        # Get video text from URL
        video_text = get_transcript(youtube_url)

        if video_text.startswith("Error"):
            st.error(video_text)
        else:
            # Perform advanced NLP analysis using NLTK
            top_entities, top_positive_sentences, top_negative_sentences, text = advanced_nlp_analysis(video_text)

            # Display top entities
            st.subheader("Top 10 Longest Character Entities")
            for entity, label in top_entities:
                st.write(f"{entity} (Type: {label})")

            # Provide a sentiment summary
            sentiment_summary = "Positive" if TextBlob(text).sentiment.polarity > 0 else "Neutral" if TextBlob(text).sentiment.polarity == 0 else "Negative"
            st.write(f"Sentiment Summary: {sentiment_summary}")

            # Display top 5 positive sentences
            st.subheader("Top 5 Positive Sentences")
            for sentence, _ in top_positive_sentences:
                st.write(sentence)

            # Display top 5 negative sentences
            st.subheader("Top 5 Negative Sentences")
            for sentence, _ in top_negative_sentences:
                st.write(sentence)

            # Download option for the text
            st.subheader("Download Video Text")
            st.write("Click below to download the video text as a text file.")
            st.download_button(
                label="Download Text",
                data=text,
                file_name="video_text.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
