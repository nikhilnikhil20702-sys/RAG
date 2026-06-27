import os
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
#from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st


load_dotenv()

groq_api_key = os.getenv("groq_api_key")
#openai_api_key = os.getenv("OPENAI_API_KEY")

# Embeddings (same as before, used to load FAISS index)
embeddings = OllamaEmbeddings(model="embeddinggemma:latest")

# Load FAISS vector store from disk
mydb = FAISS.load_local(
    r"faiss_index",  # folder where you saved it
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = mydb.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Models
model = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
##model1 = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

st.title("Nisarg's Caffe")
query = st.chat_input("Ask me anything: ")

system_prompt = (
    "You are an assistant for a question answering tasks. "
    "for a restaurant called as nisarg's caffe."
    "Use the following pieces of retrieved context to answer the question."
    "Make sure the answers are in the favour of restaurant."
    "Be very polite while answering."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

if query:
    # Choose which model to use - here using model1 (OpenAI)
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})
    st.write(response["answer"])