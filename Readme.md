# GenInsight: Generative AI Analyzer

Welcome to **GenInsight**, a generative AI-driven web application designed to analyze news articles and provide insightful answers to user queries, particularly focused on market trends and financial research. Built with Python, Streamlit, and LangChain, this project leverages advanced AI models to process real-time news data and deliver context-aware responses with sourced references.

## Project Overview

- **Purpose**: Extracts and analyzes news articles from provided URLs, answering questions using generative AI.
- **Technologies**: Python, Streamlit, Transformers (`flan-t5-base`), SentenceTransformers, LangChain, FAISS.
- **Features**: Processes news URLs, generates answers with sources, and handles market-related queries efficiently.

## Prerequisites

- **Python**: Version 3.8 or higher.
- **pip**: Python package manager (usually included with Python).
- **Git**: For cloning the repository (optional if downloading manually).

## Installation

Follow these steps to set up and run the GenInsight project on your local machine:

1. **Clone the Repository**:
   - Open a terminal and run:
     ```bash
     git clone https://github.com/your-username/geninsight.git
     cd geninsight
     Replace your-username with your GitHub username and geninsight with your repository name.

2. **Create a Virtual Environment (Recommended):**
   Create a virtual environment to isolate dependencies:
     python -m venv .venv
      Activate it:
      On Windows:
         .venv\Scripts\activate
      On macOS/Linux:
          source .venv/bin/activate

3. **Install Dependencies:**
   Install the required Python libraries listed in requirements.txt:
      pip install -r requirements.txt
   Ensure all dependencies are installed without errors.

4. **Verify Installation:**
   Check that Python and the libraries are correctly installed by running:
       python -c "import streamlit, transformers, sentence_transformers, langchain, faiss; print('All libraries loaded successfully!')"
   If no errors appear, you’re ready to proceed.



**Running the Application**
1. **Launch the Streamlit App:**
   In the terminal (with the virtual environment activated), run:
      streamlit run main.py --server.fileWatcherType none
   The --server.fileWatcherType none flag disables file watching to avoid runtime warnings with PyTorch.

2. **Interact with GenInsight:**
   The app will open in your default web browser at http://localhost:8501.
   In the sidebar, enter up to 3 news article URLs (e.g., financial news sites).
   Click "Process URLs" to load and index the articles.
   Enter a question (e.g., "What is the latest update on the stock market today?") in the main panel and press Enter to get an answer with sources.
