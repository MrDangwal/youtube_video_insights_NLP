import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from textblob import TextBlob
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
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
    sentiment_score = TextBlob(text).sentiment.polarity

    # Named Entity Recognition using NLTK
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    chunked_words = ne_chunk(tagged_words)

    for subtree in chunked_words:
        if isinstance(subtree, Tree):
            entity = " ".join([word for word, pos in subtree.leaves()])
            entities.append((entity, subtree.label()))

    # Additional sentiment analysis using TextBlob
    sentences_with_sentiments = [(sentence, analyze_sentiment(sentence)) for sentence in nltk.sent_tokenize(text)]
    top_emotional_sentences = sorted(sentences_with_sentiments, key=lambda x: abs(x[1]), reverse=True)[:10]
    top_negative_sentences = [sent for sent in top_emotional_sentences if sent[1] < 0][:5]
    top_positive_sentences = [sent for sent in top_emotional_sentences if sent[1] > 0][:5]

    # Get the top 10 entities with the longest characters
    top_entities_longest = sorted(entities, key=lambda x: len(x[0]), reverse=True)[:10]

    # Download option for the text
    st.download_button("Download Video Text", data=text, file_name="video_text.txt")

    return top_entities_longest, top_negative_sentences, top_positive_sentences

def print_summary(title, values):
    summary = f"{title}:\n"
    for idx, value in enumerate(values, start=1):
        if isinstance(value, list):
            for sub_idx, sub_value in enumerate(value, start=1):
                summary += f"  {idx}.{sub_idx}. {sub_value[0]} (Sentiment: {sub_value[1]:.2f})\n"
        else:
            summary += f"  {idx}. {value[0]} (Type: {value[1]})\n"
    return summary

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
            top_entities_longest, top_negative_sentences, top_positive_sentences = advanced_nlp_analysis(video_text)

            # Display analysis results
            st.subheader("Top 10 Entities with Longest Characters")
            st.write(print_summary("Top 10 Entities with Longest Characters", top_entities_longest))

            st.subheader("Top 5 Negative Sentences")
            st.write(print_summary("Top 5 Negative Sentences", top_negative_sentences))

            st.subheader("Top 5 Positive Sentences")
            st.write(print_summary("Top 5 Positive Sentences", top_positive_sentences))

if __name__ == "__main__":
    main()
