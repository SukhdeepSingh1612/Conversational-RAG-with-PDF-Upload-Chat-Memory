# Conversational RAG with PDF Upload and Chat History

This project implements a Conversational Retrieval-Augmented Generation (RAG) system using **Streamlit** and **LangChain**. It allows users to upload multiple PDFs, chat with the assistant, and get context-aware answers based on the content of the uploaded documents.

The system uses **Groq** for the LLM and **HuggingFace Embeddings** for document embeddings. It handles multiple PDFs, stores chat history, and answers questions based on the content and context of the uploaded documents.

## Features
- **Upload Multiple PDFs:** Upload one or more PDFs for the assistant to learn from.
- **Context-Aware Q&A:** The assistant answers questions based on the content of the uploaded PDFs.
- **Chat History:** The assistant retains chat history for the current session to provide more accurate answers.
- **Groq API Integration:** The system integrates with the Groq API for large language models (LLMs).
- **HuggingFace Embeddings:** The system utilizes **HuggingFace** embeddings for document-based similarity search.
- **Interactive UI:** Built with **Streamlit**, the UI is designed to be easy to use with feedback and interactive elements.

## Prerequisites
Before running this project, make sure you have the following installed:

- Python 3.7+
- Streamlit
- LangChain
- Groq API Key (required for LLM)
- HuggingFace API Key (required for embeddings)

You can install the necessary libraries with the following command:
```
pip install streamlit langchain chromadb langchain_groq langchain_huggingface PyPDF2 dotenv
```

## Usage
1. Clone or download this repository to your local machine.
2. Ensure you have your **Groq API key** and **HuggingFace API key** ready.
3. Create a `.env` file in the project directory with the following content:
    ```
    HF_TOKEN=<Your_HuggingFace_API_Key>
    ```
4. Run the Streamlit app by using the command:
    ```
    streamlit run betterApp.py
    ```
5. Open the app in your browser and upload PDFs, enter your Groq API key, and interact with the assistant.

## Project Structure
- `app.py`: The main Streamlit application.
- `.env`: Stores the **HuggingFace** API key.
- `requirements.txt`: List of all Python dependencies.
- `README.md`: This file.

## Chat History
The assistant stores the chat history for each session. You can view the chat history in the **Chat History** section.

## License
This project is licensed under the MIT License.
