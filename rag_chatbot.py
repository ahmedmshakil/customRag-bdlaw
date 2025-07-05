import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import streamlit as st
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set your GOOGLE_API_KEY in the .env file")
    st.stop()

# Configure Google Gemini
genai.configure(api_key=GOOGLE_API_KEY)

def load_and_process_data():
    # Load the markdown file
    loader = UnstructuredMarkdownLoader("volume_roman_XIII.md")
    documents = loader.load()
    
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./data"
    )
    
    return vectorstore

def initialize_chain(vectorstore):
    try:
        # Initialize Gemini Pro model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.0-pro",  # Updated model name
            temperature=0.7,
            convert_system_message_to_human=True,
            safety_settings={
                "HARASSMENT": "block_none",
                "HATE_SPEECH": "block_none",
                "SEXUALLY_EXPLICIT": "block_none",
                "DANGEROUS_CONTENT": "block_none",
            }
        )
    except Exception as e:
        st.error(f"Error initializing Gemini model. Please check your API key. Error: {str(e)}")
        st.stop()
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create the chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    
    return chain

# Streamlit UI
st.title("Bangladesh Law Chatbot (Powered by Google Gemini)")
st.write("Ask questions about Bangladesh law based on Volume XIII")

# Initialize session state
if "chain" not in st.session_state:
    with st.spinner("Loading and processing the law document... This might take a few minutes."):
        try:
            vectorstore = load_and_process_data()
            st.session_state.chain = initialize_chain(vectorstore)
            st.success("Document processing completed! You can now ask questions.")
        except Exception as e:
            st.error(f"Error during initialization: {str(e)}")
            st.stop()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question about Bangladesh law"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = st.session_state.chain({"question": prompt})
                ai_response = response["answer"]
                st.markdown(ai_response)
                
                # Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

# Add a sidebar with information
with st.sidebar:
    st.markdown("""
    ### About this Chatbot
    This chatbot uses RAG (Retrieval-Augmented Generation) with Google's Gemini Pro model to provide accurate information about Bangladesh law based on Volume XIII.
    
    ### How to use
    - Simply type your question about Bangladesh law in the chat input
    - The bot will search through the legal document and provide relevant answers
    - The responses are generated using both the document content and AI understanding
    
    ### Note
    The responses are based on the content of Volume XIII and may not include recent updates or amendments to the law.
    """)