# Bangladesh Law Chatbot (RAG-based)

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Bangladesh law using Volume XIII as its knowledge base. Built with LangChain, Google's Gemini Pro, and Streamlit.

## Quick Start

```bash
# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your Google API key to .env file
# GOOGLE_API_KEY=your_key_here

# Run the app
streamlit run rag_chatbot.py
```

## Features

- 📚 RAG-based QA system using Volume XIII of Bangladesh Law
- 🔍 Local embeddings with sentence-transformers
- 💾 Vector storage with ChromaDB
- 🤖 Gemini Pro for response generation
- 🌐 Streamlit web interface

## Project Structure

```
.
├── rag_chatbot.py    # Main application
├── requirements.txt   # Dependencies
├── .env              # Environment variables
└── volume_roman_XIII.md  # Source document
```

## Requirements

- Python 3.8+
- Google API key (Get from [Google MakerSuite](https://makersuite.google.com/app/apikey))
- 2GB+ RAM recommended

## Environment Variables

```env
GOOGLE_API_KEY=your_google_api_key_here
```

## Tech Stack

- LangChain - RAG pipeline
- Sentence Transformers - Text embeddings
- ChromaDB - Vector store
- Google Gemini Pro - LLM
- Streamlit - Web interface

## Author 

Shakil Ahmed

- Project is unfinished due to paid api Key