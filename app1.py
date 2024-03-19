import os
import pygame
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from googletrans import LANGUAGES, Translator
from txtai.pipeline import Summary
from rake_nltk import Rake

translator = Translator()  # Initialize the translator module.
pygame.mixer.init()  # Initialize the mixer module.

isTranslateOn = False  # Initialize the translation status variable

def summarize_text(text):
    """
    Summarizes the provided text using a pre-trained summarization model.
    Args:
        text (str): The text to be summarized.
    Returns:
        str: The summarized text.
    """
    # Create summary instance
    summary = Summary("sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")
    result = summary(text)
    return result

# Create a mapping between language names and language codes
language_mapping = {name: code for code, name in LANGUAGES.items()}

def get_language_code(language_name):
    return language_mapping.get(language_name, language_name)

def translator_function(text_data, from_language, to_language):
    try:
        translated_text = translator.translate(text_data, src=from_language, dest=to_language)
        return translated_text.text
    except Exception as e:
        return str(e)

def text_to_voice(text_data, to_language):
    try:
        tts = gTTS(text=text_data, lang=to_language, slow=False)
        tts.save("cache_file.mp3")
        audio = pygame.mixer.Sound("cache_file.mp3")  # Load a sound.
        audio.play()
        os.remove("cache_file.mp3")
    except Exception as e:
        st.error(f"Error occurred while converting text to voice: {e}")

def main_process(output_placeholder, from_language, to_language):
    global isTranslateOn

    while isTranslateOn:
        rec = sr.Recognizer()
        with sr.Microphone() as source:
            output_placeholder.text("Listening...")
            rec.pause_threshold = 1
            audio = rec.listen(source, phrase_time_limit=10)

        try:
            output_placeholder.text("Processing...")
            spoken_text = rec.recognize_google(audio, language=from_language)

            output_placeholder.text("Translating...")
            translated_text = translator_function(spoken_text, from_language, to_language)
            text_to_voice(translated_text, to_language)

        except Exception as e:
            st.error(f"Error occurred during translation: {e}")

# UI layout
st.title("Multifunctional NLP Application")

# Sidebar for choosing functionality
selected_functionality = st.sidebar.radio("Select Functionality:", ["Translation", "Summarization", "Keyword Extraction"])

if selected_functionality == "Translation":
    st.title("Language Translator")
    # Dropdowns for selecting languages
    from_language_name = st.selectbox("Select Source Language:", list(LANGUAGES.values()))
    to_language_name = st.selectbox("Select Target Language:", list(LANGUAGES.values()))

    # Swap button
    if st.button("Swap Language"):
        from_language_name, to_language_name = to_language_name, from_language_name

    # Convert language names to language codes
    from_language = get_language_code(from_language_name)
    to_language = get_language_code(to_language_name)

    # Unique key for text input
    text_input_key = "text_input_1"

    # Text input for user to type
    text_input = st.text_area("Type your text:", key=text_input_key)

    # Button to trigger translation
    start_button = st.button("Start Translation")
    stop_button = st.button("Stop Translation")
    listen_button = st.button("Listen to Translation")

    # Check if "Start" button is clicked
    if start_button:
        if not isTranslateOn:
            isTranslateOn = True
            output_placeholder = st.empty()
            main_process(output_placeholder, from_language, to_language)

    # Check if "Stop" button is clicked
    if stop_button:
        isTranslateOn = False

    # Check if "Listen" button is clicked
    if listen_button and text_input:
        translated_text = translator_function(text_input, from_language, to_language)
        text_to_voice(translated_text, to_language)

    # Button to trigger text translation
    translate_button = st.button("Translate")

    # Check if "Translate" button is clicked
    if translate_button and text_input:
        translated_text = translator_function(text_input, from_language, to_language)
        st.write("Translated Text:")
        st.write(translated_text)

elif selected_functionality == "Summarization":
    st.title("Text Summarization")
    input_text = st.text_area("Type or paste your text here:")
    summarize_button = st.button("Summarize")

    if summarize_button:
        st.write("Generating summary... Please wait.")

        try:
            summarized_text = summarize_text(input_text)
            st.success("Summary:")
            st.write(summarized_text)
        except Exception as e:
            st.error(f"An error occurred while summarizing: {e}")

elif selected_functionality == "Keyword Extraction":
    st.title("Keyword Extraction")
    input_text = st.text_area("Paste your job description here:")
    extract_button = st.button("Extract Keywords")
    
    if extract_button:
        st.write("Extracting keywords... Please wait.")
        
        # Perform keyword extraction using Rake
        rake = Rake()
        rake.extract_keywords_from_text(input_text)
        keywords = rake.get_ranked_phrases_with_scores()
        
        # Filter keywords based on relevance
        relevant_keywords = [keyword for score, keyword in keywords if score > 1.0]
        
        st.write("Keywords Extracted:")
        st.write(relevant_keywords)
