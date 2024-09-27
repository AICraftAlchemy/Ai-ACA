import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
import re
import base64
import time
import random

# Load environment variables
load_dotenv()

def clean_text(text):
    text = re.sub(r'<[^>]*?>', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$$\$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()
    text = ' '.join(text.split())
    return text

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.2-3b-preview")

    def answer_question(self, question):
        prompt_question = PromptTemplate.from_template(
            """
            ### QUESTION:
            {question}

            ### INSTRUCTION:
            Answer the question thoroughly based on your knowledge.
            Provide an informative and accurate answer.

            """
        )
        chain_question = prompt_question | self.llm
        res = chain_question.invoke(input={"question": question})
        return res.content.strip()

def set_theme(is_dark_mode):
    if is_dark_mode:
        st.markdown("""
        <style>
        :root {
            --background-color: #121212;
            --text-color: #E0E0E0;
            --card-background: #1F1F1F;
            --button-color: #BB86FC;
            --button-hover: #3700B3;
            --accent-color: #03DAC6;
            --secondary-color: #03A9F4;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        :root {
            --background-color: #FFFFFF;
            --text-color: #333333;
            --card-background: #F9F9F9;
            --button-color: #03A9F4;
            --button-hover: #0288D1;
            --accent-color: #FF4081;
            --secondary-color: #4CAF50;
        }
        </style>
        """, unsafe_allow_html=True)

def create_streamlit_app(llm):
    st.set_page_config(page_title="AI ACA", page_icon="💡", layout="wide")

    # Initialize session state for theme
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = True

    # Apply theme
    set_theme(st.session_state.dark_mode)

    # Custom CSS for the app
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        body {
            font-family: 'Poppins', sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            transition: all 0.3s ease;
        }
        .stApp {
            background-color: var(--background-color);
        }
        .main {
            background-color: var(--background-color);
        }
        .title {
            text-align: center;
            color: var(--text-color);
            font-size: 3.5em;
            font-weight: 700;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            background: linear-gradient(45deg, var(--button-color), var(--accent-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            color: var(--text-color);
            font-size: 1.5em;
            margin-bottom: 30px;
            font-weight: 300;
        }
        .container {
            padding: 30px;
            max-width: 1000px;
            margin: 0 auto;
            background-color: var(--card-background);
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        .input-box {
            margin-bottom: 25px;
        }
        .input-box input, .input-box textarea {
            width: 100%;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background-color: var(--background-color);
            color: var(--text-color);
            font-size: 16px;
            transition: all 0.3s ease;
        }
        .input-box input:focus, .input-box textarea:focus {
            outline: none;
            box-shadow: 0 0 0 2px var(--accent-color);
        }
        .stButton > button {
            background: linear-gradient(45deg, var(--button-color), var(--accent-color));
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            font-size: 18px;
            font-weight: 700;
            transition: all 0.3s ease;
            width: 100%;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        }
        .result-card {
            background-color: var(--card-background);
            padding: 25px;
            border-radius: 15px;
            margin-top: 30px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        .result-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
        }
        .result-card h3 {
            color: var(--accent-color);
            border-bottom: 2px solid var(--accent-color);
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-weight: 700;
        }
        .footer {
            text-align: center;
            color: var(--text-color);
            font-size: 1em;
            margin-top: 40px;
            padding: 20px;
            background-color: var(--card-background);
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .footer:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        }
        .footer a {
            color: var(--accent-color);
            text-decoration: none;
            transition: all 0.3s ease;
            font-weight: 700;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        .loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 24px;
            font-weight: 700;
            color: var(--text-color);
        }
        .animation {
            animation: animate 2s infinite;
        }
        @keyframes animate {
            0% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.1);
            }
            100% {
                transform: scale(1);
            }
        }
    </style>
    """, unsafe_allow_html=True)

    # Theme toggle
    col1, col2 = st.columns([4, 1])
    with col2:
        st.markdown("""
        <div class="toggle-container">
            <span class="toggle-label">Dark Mode</span>
            <label class="toggle-switch">
                <input type="checkbox" id="theme-toggle">
                <span class="toggle-slider"></span>
            </label>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Toggle Theme", key="theme_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.experimental_rerun()

    st.markdown("<div class='title'>💡 AI ACA</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Ask any question and get an AI-powered answer.</div>", unsafe_allow_html=True)

    # Detect Enter key for submission
    question = st.text_area("Type your question below", height=150, key="question_input")
    if st.button("Submit") or (st.session_state.get("question_input") and st.session_state.get("question_input").strip()):
        with st.spinner("Processing..."):
            st.write("🤔 Analyzing your question...")
            time.sleep(1)
            st.write("🔍 Searching for relevant information...")
            time.sleep(1)
            st.write("💡 Generating answer...")
            time.sleep(1)
            answer = llm.answer_question(question)
            st.write("🎉 Answer generated!")
            st.markdown(f"<div class='result-card'><h3>Answer:</h3><p>{answer}</p></div>", unsafe_allow_html=True)

    st.markdown("<div class='footer'>Built with 💖 by <a href='#'>Ai Craft Alchemy  | Contact :+91 7661081043</a></div>", unsafe_allow_html=True)

    # Animation
    st.markdown("<div class='animation'>🤖</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    chain = Chain()
    create_streamlit_app(chain)
