# AmbedkarGPT-Intern-Task

This is a submission for the Kalpit Pvt Ltd AI Intern Hiring Assignment.
It is a simple command-line Q&A system built using Python and the LangChain framework.

The system will ingest the text from a provided short speech by Dr. B.R. Ambedkar (`speech.txt`) and answer questions based solely on that content.

## Features

* **RAG Pipeline:** Uses a Retrieval-Augmented Generation (RAG) pipeline to provide context-aware answers.
* **Local-First:** All components run 100% locally.
* **LLM:** Uses **Ollama** with **Mistral 7B**.
* **Embeddings:** Uses **HuggingFace** `sentence-transformers/all-MiniLM-L6-v2`.
* **Vector Store:** Uses **ChromaDB** for local vector storage.
* **Framework:** Built with **LangChain**.

## How to Set Up and Run

### 1. Prerequisites
* Python 3.8+ 
* [Ollama](https://ollama.ai/) installed

### 2. Setup

1.  **Clone the Repository:**
    (Remember to replace `[YOUR_GITHUB_USERNAME]` with your actual username)
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_GITHUB_USERNAME]/AmbedkarGPT-Intern-Task.git
    cd AmbedkarGPT-Intern-Task
    ```

2.  **Create a Python Environment:**
    (Using `conda`)
    ```bash
    conda create -n ambedkar-qa python=3.10
    conda activate ambedkar-qa
    ```
    (Or using `venv`)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Install all required packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Pull the Ollama Model:**
    Make sure Ollama is running, then pull the `mistral` model.
    ```bash
    ollama pull mistral
    ```

### 3. Run the System

With your Ollama application running in the background, execute the main script:

```bash
python main.py
