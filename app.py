import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import re

# Load environment variables
load_dotenv()

def clean_text(text):
    text = re.sub(r'<[^>]*?>', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class LlamaAIChain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0.7, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")
        self.memory = ConversationBufferMemory(return_messages=True)

    def ask_question(self, question):
        prompt = PromptTemplate(
            input_variables=["history", "question"],
            template="Chat History:\n{history}\nHuman: {question}\n\nAI: Let me think about that and provide a helpful response."
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, memory=self.memory)
        response = chain.run(question=question)
        return response

    def analyze_website(self, url, question):
        loader = WebBaseLoader([url])
        data = clean_text(loader.load()[0].page_content)
        
        prompt = PromptTemplate(
            input_variables=["website_content", "question"],
            template="""
            Analyze the following website content and answer the user's question:

            Website Content:
            {website_content}

            User's Question:
            {question}

            Provide a detailed and informative answer based on the website content:
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(website_content=data, question=question)
        return response

def set_page_config():
    st.set_page_config(page_title="Llama AI Platform", page_icon="🦙", layout="wide")
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        font-family: Arial, sans-serif;
    }
    .main-title {
        color: #4a4a4a;
        text-align: center;
        margin-bottom: 30px;
    }
    .section-title {
        color: #2c3e50;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .response-area {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .stTextInput>div>div>textarea {
        min-height: 100px;
    }
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e
    }
    .chat-message.bot {
        background-color: #475063
    }
    .chat-message .avatar {
      width: 20%;
    }
    .chat-message .avatar img {
      max-width: 78px;
      max-height: 78px;
      border-radius: 50%;
      object-fit: cover;
    }
    .chat-message .message {
      width: 80%;
      padding: 0 1.5rem;
      color: #fff;
    }
    .toggle-button {
        position: absolute;
        top: 10px;
        left: 100px;
        z-index: 1000;
    }
    .footer {
        text-align: center;
        padding: 20px 0;
        font-size: 14px;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 40px;
    }
    </style>
    """, unsafe_allow_html=True)

def chat_interface():
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Enter your question:")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.llama_chain.ask_question(question)
                st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

def website_analysis_interface():
    url = st.text_input("Enter website URL:")
    website_question = st.text_area("Enter your question about the website:", height=100, key="website_question_input")
    if st.button("Analyze"):
        if url and website_question:
            with st.spinner("Analyzing website..."):
                analysis = st.session_state.llama_chain.analyze_website(url, website_question)
                st.markdown("<div class='response-area'>", unsafe_allow_html=True)
                st.write("Analysis:", analysis)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Please enter both a URL and a question.")

def create_streamlit_app():
    set_page_config()
    
    if 'llama_chain' not in st.session_state:
        st.session_state.llama_chain = LlamaAIChain()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'current_interface' not in st.session_state:
        st.session_state.current_interface = "chat"

    # Toggle button at the top left
    st.markdown("<div class='toggle-button'>", unsafe_allow_html=True)
    button_label = "Switch to Web Analyzer" if st.session_state.current_interface == "chat" else "Switch to Chat with AI"
    if st.button(button_label):
        st.session_state.current_interface = "website" if st.session_state.current_interface == "chat" else "chat"
        st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<h1 class='main-title'>🦙 Llama AI Platform by Ai Craft Alchemy</h1>", unsafe_allow_html=True)

    # Main content area
    if st.session_state.current_interface == "chat":
        st.markdown("<h2 class='section-title'>Interact with AI ACA</h2>", unsafe_allow_html=True)
        chat_interface()
    else:
        st.markdown("<h2 class='section-title'>Analyze Website</h2>", unsafe_allow_html=True)
        website_analysis_interface()

    # Add footer
    st.markdown("""
    <div class='footer'>
    Developed  by  <a href='https://aicraftalchemy.github.io'>Ai Craft Alchemy</a><br>
    Connect with us: <a href='tel:+917661081043'>+91 7661081043</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    create_streamlit_app()
