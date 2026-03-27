import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import google.generativeai as genai
from langchain_core.prompts import PromptTemplate

# ------------------- CONFIG -------------------
st.set_page_config(page_title="ReadBetween.AI", page_icon="👀")
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #ffffff;
}

.block-container {
    padding-top: 2rem;
    max-width: 800px;
    margin: auto;
}

.stChatMessage {
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 10px;
    background-color: #1a1d24;
}

.stTextInput input {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align: center;'>👀 ReadBetween.AI</h1>
<p style='text-align: center; color: gray;'>Because your PDF hides things.</p>
""", unsafe_allow_html=True)

# ------------------- GEMINI SETUP -------------------
genai.configure(api_key="AIzaSyDNcAHSVgFur_w5pGkFCUOGw_fFIVvwIPU")

# ------------------- FILE UPLOAD -------------------
st.markdown("""
<div style="
    background-color: #1a1d24;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
">
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

st.markdown("</div>", unsafe_allow_html=True)

if not uploaded_file:
    st.info("👆 Upload a PDF to get started")
    st.stop()

# ------------------- LOAD PDF -------------------
with open("temp.pdf", "wb") as f:
    f.write(uploaded_file.read())

loader = PyPDFLoader("temp.pdf")
documents = loader.load()

# ------------------- SPLIT TEXT -------------------
text_splitter = CharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=50
)
texts = text_splitter.split_documents(documents)

# ------------------- EMBEDDINGS -------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ------------------- STORE DB (SMART CACHE) -------------------
if "file_name" not in st.session_state or st.session_state.file_name != uploaded_file.name:
    st.session_state.db = FAISS.from_documents(texts, embeddings)
    st.session_state.file_name = uploaded_file.name

db = st.session_state.db

# ------------------- CHAT MEMORY -------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------- USER QUERY -------------------
query = st.chat_input("Ask something from your PDF...")

if query:
    docs = db.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = PromptTemplate.from_template(
        """
        Answer ONLY using the context below.
        If not found, say "Not found in document".

        Context:
        {context}

        Question:
        {question}
        """
    )

    final_prompt = prompt.format(context=context, question=query)

    # ------------------- GEMINI RESPONSE -------------------
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(final_prompt).text

    # Smart fallback
    if "not found" in response.lower():
        response = "⚠️ Not found in the document."

    # Store chat
    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("assistant", response))

# ------------------- DISPLAY CHAT -------------------
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# ------------------- SOURCES -------------------
if query:
    st.subheader("🔍 Sources")
    for doc in docs:
        page = doc.metadata.get("page", 0)
        st.markdown(f"👀 Found between lines on **page {page + 1}**...")
        st.markdown(f"""
<div style="background-color:#1a1d24; padding:10px; border-radius:10px; margin-bottom:10px;">
{doc.page_content[:200]}
</div>
""", unsafe_allow_html=True)
    
