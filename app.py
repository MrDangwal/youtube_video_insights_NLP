import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import string


nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

def get_transcript(youtube_url):
    try:
        video_id = YouTube(youtube_url).video_id
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ' '.join([entry['text'] for entry in transcript])
        return text
    except Exception as e:
        return f"Error: {e}"

def clean_text(text):
    
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def split_text_into_sentences(text, max_words_per_sentence):
    sentences = []
    words = text.split()
    current_sentence = []

    for word in words:
        if len(current_sentence) + len(word.split()) <= max_words_per_sentence:
            current_sentence.extend(word.split())
        else:
            sentences.append(" ".join(current_sentence))
            current_sentence = [word]

    if current_sentence:
        sentences.append(" ".join(current_sentence))

    return sentences

def advanced_nlp_analysis(text):
    entities = []
    positive_sentences = []
    negative_sentences = []

    
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    chunked_words = ne_chunk(tagged_words)

    for subtree in chunked_words:
        if isinstance(subtree, Tree):
            entity = " ".join([word for word, pos in subtree.leaves()])
            entities.append((entity, subtree.label()))

    
    entities.sort(key=lambda x: len(x[0]), reverse=True)
    top_entities = entities[:10]

    
    max_words_per_sentence = 20
    sentences = split_text_into_sentences(text, max_words_per_sentence)

    
    sid = SentimentIntensityAnalyzer()
    for sentence in sentences:
        cleaned_sentence = clean_text(sentence)
        sentiment_score = sid.polarity_scores(cleaned_sentence)['compound']

        if sentiment_score > 0.05:
            positive_sentences.append((sentence, sentiment_score))
        elif sentiment_score < -0.05:
            negative_sentences.append((sentence, sentiment_score))

    
    positive_sentences.sort(key=lambda x: x[1], reverse=True)
    negative_sentences.sort(key=lambda x: x[1])

    return top_entities, positive_sentences[:5], negative_sentences[:5], text

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

    
    youtube_url = st.text_input("Enter YouTube Video URL")
    if st.button("Analyze"):
        st.write("Analyzing...")

        
        video_text = get_transcript(youtube_url)

        if video_text.startswith("Error"):
            st.error(video_text)
        else:
            
            top_entities, positive_sentences, negative_sentences, text = advanced_nlp_analysis(video_text)

            
            st.subheader("Top 10 entities mentioned in the Video")
            for entity, label in top_entities:
                st.write(f"{entity} (Type: {label})")

            
            st.subheader("Top 5 Positive Sentences")
            for sentence, score in positive_sentences:
                st.write(f"{sentence} (Sentiment: {score:.2f})")

            
            st.subheader("Top 5 Negative Sentences")
            for sentence, score in negative_sentences:
                st.write(f"{sentence} (Sentiment: {score:.2f})")

            
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
