### RAG Q&A Conversation With PDF Including Chat History
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


os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")


## Set up streamlit app
st.title("Conversational RAG with PDF upload and chat hostory")
st.write("Upload Pdf's and chat with their content")

## Input trhe Groq API key
api_key =st.text_input("Enter your Groq API key :", type="password")


## Check if groq api key is provided
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")


    ## Chat interface

    session_id = st.text_input("Session ID", value="default_session")
    ## Statefully manage the chat hisroty

    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
    ## process upload PDF's
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.read())
                file_name = uploaded_file.name
            
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)



        ## Split and create embeddings for the documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap=500)
            split_docs = text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(split_docs, embeddings)
            retriever = vectorstore.as_retriever()


        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question" \
            "which might reference context in the chat history," \
            "formulate a standalone question which can be understoor" \
            "without the chat history. Do NOT anserr the question," \
            "just reformulate it if needed and otherwise return it as is."
        )

        contexualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm,retriever,contexualize_q_prompt)

        ## Answer question

        system_prompt = (
            "You are an assistant for a question-answering tasks." \
            "Use the following pieces of retrieved context to answer" \
            "the questions. If you don't know the answer, say that you don't know." \
            "Use three sentences maximum and keep answer concise." \
            "\n\n" \
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )


        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                }, # contructs a key "abc123" in 'store'
            )
            st.write(st.session_state.store)
            st.success(f"Assistant: {response['answer']}")
            st.write("Chat History", session_history.messages)


else: 
    st.warning("Please enter Groq API key")
















