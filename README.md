# AmbedkarGPT-Intern-Task

[cite_start]This is a submission for the Kalpit Pvt Ltd AI Intern Hiring Assignment[cite: 1].
[cite_start]It is a simple command-line Q&A system built using Python and the LangChain framework[cite: 4, 15].

The system will ingest the text from a provided short speech by Dr. B.R. [cite_start]Ambedkar (`speech.txt`) and answer questions based solely on that content[cite: 5, 6].

## Features

* [cite_start]**RAG Pipeline:** Uses a Retrieval-Augmented Generation (RAG) pipeline to provide context-aware answers[cite: 7].
* **Local-First:** All components run 100% locally.
* [cite_start]**LLM:** Uses **Ollama** with **Mistral 7B**[cite: 18].
* [cite_start]**Embeddings:** Uses **HuggingFace** `sentence-transformers/all-MiniLM-L6-v2`[cite: 17].
* [cite_start]**Vector Store:** Uses **ChromaDB** for local vector storage[cite: 16].
* [cite_start]**Framework:** Built with **LangChain**[cite: 15].

## How to Set Up and Run

### 1. Prerequisites
* [cite_start]Python 3.8+ [cite: 14]
* [cite_start][Ollama](https://ollama.ai/) installed[cite: 45].

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
    [cite_start]Install all required packages from the `requirements.txt` file[cite: 21].
    ```bash
    pip install -r requirements.txt
    ```

4.  **Pull the Ollama Model:**
    [cite_start]Make sure Ollama is running, then pull the `mistral` model[cite: 45, 58].
    ```bash
    ollama pull mistral
    ```

### 3. Run the System

[cite_start]With your Ollama application running in the background, execute the main script[cite: 20]:

```bash
python main.py
