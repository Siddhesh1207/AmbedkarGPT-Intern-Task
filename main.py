import sys
# Core LangChain imports
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Community package imports (LLMs, Embeddings, Loaders, VectorStores)
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# Text splitter import
from langchain_text_splitters import CharacterTextSplitter

# --- 1. Load the Document ---
# Load the 'speech.txt' file
try:
    loader = TextLoader("speech.txt", encoding="utf-8")
    documents = loader.load()
except FileNotFoundError:
    print("Error: 'speech.txt' file not found.")
    print("Please make sure the file is in the same directory as main.py")
    sys.exit(1)
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit(1)

# --- 2. Split the Text ---
# Split the document into manageable chunks
text_splitter = CharacterTextSplitter(
    separator="\n",  # Split on new lines
    chunk_size=200,  # Aim for chunks of this size
    chunk_overlap=50, # Overlap chunks to maintain context
    length_function=len
)
texts = text_splitter.split_documents(documents)

# --- 3. Create Embeddings ---
# Use the specified HuggingFace embedding model 
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- 4. Store in Vector Database ---
# Use ChromaDB as the local vector store 
db = Chroma.from_documents(texts, embeddings)

# --- 5. Setup the LLM ---
# Use the local Ollama LLM with Mistral
llm = Ollama(model="mistral")

# --- 6. Create the RAG Chain ---
# This is the new (v1.0+) way to create a RAG chain
print("Creating RAG chain...")

# 6a. Create a Prompt Template
# This defines how the context and question are passed to the LLM
prompt = ChatPromptTemplate.from_template(
    """Answer the user's question based only on the following context:

<context>
{context}
</context>

Question: {input}"""
)

# 6b. Create the "Stuff" Documents Chain
# This chain takes the context and question, formats them using the prompt, and sends the result to the LLM. 
document_chain = create_stuff_documents_chain(llm, prompt)

# 6c. Create the Retriever
# This object is responsible for fetching relevant documents from the vector store (Chroma). 
retriever = db.as_retriever()

# 6d. Create the Retrieval Chain
# This is the final chain. It takes the user's input (question),
# passes it to the retriever to get documents, and then passes the
# original input and the retrieved documents to the document_chain.
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- 7. Run the Q&A System ---
print("-------------------------------------------------")
print("  Ambedkar-GPT Q&A System Initialized  ")
print("-------------------------------------------------")
print("Ask a question based on the 'speech.txt' content.")
print("Type 'exit' to quit.\n")

# Simple command-line loop
while True:
    try:
        query = input("Ask a question: ")
        if query.lower() == 'exit':
            print("Exiting...")
            break
        
        if not query.strip():
            print("Please enter a question.")
            continue

        # Run the chain and get the result
        # The input key must match the prompt template (e.g., "input")
        response = retrieval_chain.invoke({"input": query})
        
        # Print the answer
        # The response dictionary now contains the key 'answer'
        print("\nAnswer:")
        print(response['answer'])
        print("-----------------\n")

    except EOFError:
        break
    except KeyboardInterrupt:
        print("\nExiting...")

        break
