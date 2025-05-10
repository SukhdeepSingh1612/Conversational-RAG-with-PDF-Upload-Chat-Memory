### Conversational RAG Q&A With PDF Upload + Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Set HuggingFace token
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# App Title
st.set_page_config(page_title="PDF Chat with Memory", layout="wide")
st.title("📄 Conversational RAG with PDF Upload + Chat Memory")
st.markdown("Upload one or more **PDFs**, ask questions, and your assistant will remember the context!")

# Groq API Key Input
api_key = st.text_input("🔑 Enter your Groq API Key:", type="password")

# Only continue if API key is provided
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")

    # Session ID
    session_id = st.text_input("🪪 Session ID", value="default_session")

    # Initialize session store
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # File uploader
    uploaded_files = st.file_uploader("📂 Upload PDF file(s)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("📚 Processing PDFs..."):
            documents = []

            for uploaded_file in uploaded_files:
                file_path = f"./temp_{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)

            # Split and embed
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            split_docs = text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(split_docs, embeddings)
            retriever = vectorstore.as_retriever()

            # Prompt for context-aware question reformulation
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed."
            )

            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])

            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt)

            # Prompt for answering questions
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following retrieved context to answer the questions. "
                "If you don't know the answer, say you don't know. "
                "Keep your response to three sentences.\n\n{context}"
            )

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            def get_session_history(session: str) -> BaseChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id] = ChatMessageHistory()
                return st.session_state.store[session_id]

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain, get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            # Chat interface
            st.markdown("---")
            user_input = st.text_input("💬 Ask a question from your uploaded PDFs:")

            if user_input:
                session_history = get_session_history(session_id)
                with st.spinner("🤖 Thinking..."):
                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": session_id}}
                    )

                st.success(f"🧠 Assistant: {response['answer']}")

                # Show formatted chat history
                with st.expander("🕘 Chat History"):
                    for msg in session_history.messages:
                        role = "🧑‍💼 You" if msg.type == "human" else "🤖 Assistant"
                        st.markdown(f"**{role}:** {msg.content}")

else:
    st.warning("Please enter your Groq API key to start.")
