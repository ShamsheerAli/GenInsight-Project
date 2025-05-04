import streamlit as st
import pickle
import time
import os
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

st.title("GenInsight:Generative AI Analyzer!")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_hf.pkl"

main_placeholder = st.empty()


# Lazy load models only when needed
@st.cache_resource
def load_models():
    model_name = "google/flan-t5-base"  # Upgraded to flan-t5-base
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = pipeline("text2text-generation", model=model_name, tokenizer=tokenizer, max_length=200,
                   num_beams=5)  # Increased max_length, added beam search
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return llm, embedding_model


# Custom embedding class for SentenceTransformer
class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

    def __call__(self, text):
        return self.embed_query(text)


if process_url_clicked:
    try:
        # Load models
        llm, embedding_model = load_models()

        # Remove duplicates from URLs
        urls = list(set(url for url in urls if url))
        if not urls:
            main_placeholder.text("No valid URLs provided.")
            st.stop()

        # Load data
        loader = UnstructuredURLLoader(urls=urls, headers={"User-Agent": "Mozilla/5.0"})
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()
        if not data or not isinstance(data, list):
            main_placeholder.text("No valid data loaded. Please check the URLs.")
            st.stop()

        # Debug: Check data structure
        #st.write("Data structure:", [type(d) for d in data][:5])

        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=500  # Reduced chunk size for better context handling
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)

        # Create embeddings and save to FAISS index
        embeddings = SentenceTransformerEmbeddings(embedding_model)
        vectorstore = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        # Save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)
    except Exception as e:
        main_placeholder.text(f"Error processing URLs: {str(e)}")
        st.stop()

query = main_placeholder.text_input("Question: ")
if query.strip():
    if os.path.exists(file_path):
        llm, embedding_model = load_models()
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            # Retrieve relevant documents
            results = vectorstore.similarity_search(query, k=3)
            context = " ".join([doc.page_content for doc in results])[:1000]  # Truncate context to 1000 chars
            # Generate answer using LLM
            input_text = f"Question: {query}\nContext: {context}\nAnswer:"  # Improved prompt format
            result = llm(input_text, num_return_sequences=1)[0]['generated_text']
            # Debug raw output
            #st.write("Raw LLM output:", result)
            # Ensure the result is meaningful text
            if not result.strip() or result.strip().replace(".", "").replace(",", "").isdigit():
                result = "Sorry, I couldn't generate a meaningful answer. Please try rephrasing the question."
            st.header("Answer")
            st.write(result)

            # Display sources
            st.subheader("Sources:")
            for doc in results:
                source = doc.metadata.get('source', 'Unknown')
                st.write(source)
    else:
        st.error("FAISS index not found. Please process URLs first.")
