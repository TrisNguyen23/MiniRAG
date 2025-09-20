import streamlit as st
from prepareData import load_and_chunk
from ragSystem import MiniRAG

st.title("Mini RAG QA System")

# Load data & initialize system
st.write("Loading data and embeddings...")
texts = load_and_chunk()
rag = MiniRAG(texts)
st.write("System ready! Ask your question.")

query = st.text_input("Ask a question:")
if query:
    answer = rag.generate_answer(query)
    st.write("**Answer:**", answer)
