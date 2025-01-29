# 🚀 Setting Up Ollama & Running DeepSeek R1 Locally for a Powerful RAG System

This guide walks you through installing [Ollama](https://ollama.ai/) and using it to run the **DeepSeek R1** model locally. We’ll integrate the local model with [LangChain](https://github.com/hwchase17/langchain) to create a **Retrieval-Augmented Generation (RAG)** system with Streamlit. 

---

## Table of Contents
1. [Introduction](#introduction)
2. [Why Run DeepSeek R1 Locally?](#why-run-deepseek-r1-locally)
3. [Step 1: Installing Ollama](#step-1-installing-ollama)
4. [Step 2: Running DeepSeek R1 on Ollama](#step-2-running-deepseek-r1-on-ollama)
5. [Step 3: Setting Up a RAG System Using Streamlit](#step-3-setting-up-a-rag-system-using-streamlit)
6. [Step 4: Running the RAG System](#step-4-running-the-rag-system)
7. [Step 5: Running the App](#step-5-running-the-app)
8. [Final Thoughts](#final-thoughts)

---

## Introduction

### 🤖 Ollama
Ollama is a framework for running large language models (LLMs) **locally** on your machine:
- **Example**: `ollama run deepseek-r1:1.5b`
- **Why use it?** Free, private, fast, and works offline.

### 🔗 LangChain
LangChain is a Python/JS framework for building AI applications that integrate LLMs with data sources, APIs, and memory:
- **Why use it?** It helps connect LLMs to real-world applications like chatbots, document Q&A, and RAG pipelines.

### 📄 RAG (Retrieval-Augmented Generation)
RAG is an AI technique that retrieves external data (e.g., PDFs, databases) and augments the LLM’s response:
- **Why use it?** Improves accuracy and reduces hallucinations by referencing actual documents.
- **Example**: AI-powered PDF Q&A system that fetches relevant document content before generating answers.

### ⚡ DeepSeek R1
DeepSeek R1 is an open-source AI model optimized for reasoning, problem-solving, and factual retrieval:
- **Why use it?** Strong logical capabilities, great for RAG applications, and can be run locally with Ollama.

### 🚀 How They Work Together
1. **Ollama** runs **DeepSeek R1** locally.  
2. **LangChain** connects the AI model to external data.  
3. **RAG** enhances responses by retrieving relevant information.  
4. **DeepSeek R1** generates high-quality answers.

---

## Why Run DeepSeek R1 Locally?

| Benefit         | Cloud-Based Models                     | Local DeepSeek R1              |
|----------------|----------------------------------------|--------------------------------|
| **Privacy**    | ❌ Data sent to external servers        | ✅ 100% Local & Secure          |
| **Speed**      | ⏳ API latency & network delays         | ⚡ Instant inference            |
| **Cost**       | 💰 Pay per API request                  | 🆓 Free after setup             |
| **Customization** | ❌ Limited fine-tuning              | ✅ Full model control           |
| **Deployment** | 🌍 Cloud-dependent                     | 🔥 Works offline & on-premises  |

---

## Step 1: Installing Ollama

1. **Go to the official Ollama download page**  
   [Download Ollama](https://ollama.ai/)  

2. **Select your operating system** (macOS, Linux, or Windows)  
3. **Click on the Download button**  
4. **Follow the system-specific instructions** to complete the installation  


---

## Step 2: Running DeepSeek R1 on Ollama

### 2.1 Pull the DeepSeek R1 Model
Pull the 1.5B parameter model:

```bash
ollama pull deepseek-r1:1.5b
```

This will download and set up the DeepSeek R1 model locally.

### 2.2 Run DeepSeek R1
Once the model is downloaded, you can run:

```bash
ollama run deepseek-r1:1.5b
```

Ollama initializes the model, and you can start interacting with it by sending queries.


---

## Step 3: Setting Up a RAG System Using Streamlit

### 3.1 Prerequisites
Before running the RAG system, ensure you have:
- Python installed  
- A Conda environment (recommended)  
- Required Python packages installed

```bash
pip install -U langchain langchain-community
pip install streamlit
pip install pdfplumber
pip install semantic-chunkers
pip install open-text-embeddings
pip install faiss
pip install ollama
pip install prompt-template
pip install langchain
pip install langchain_experimental
pip install sentence-transformers
pip install faiss-cpu
```

> **Tip:** For detailed setup, check out:  
> [Setting Up a Conda Environment for Python Projects](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

---

## Step 4: Running the RAG System

### 4.1 Clone or Create the Project
Create a new project directory:
```bash
mkdir rag-system && cd rag-system
```

### 4.2 Create a Python Script
Create a file named `app.py` and paste the following code:

```python
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

# Streamlit UI
st.title("📄 RAG System with DeepSeek R1 & Ollama")

uploaded_file = st.file_uploader("Upload your PDF file here", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()

    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)

    embedder = HuggingFaceEmbeddings()
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = Ollama(model="deepseek-r1:1.5b")

    prompt = """
    Use the following context to answer the question.
    Context: {context}
    Question: {question}
    Answer:"""

    QA_PROMPT = PromptTemplate.from_template(prompt)

    llm_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
    combine_documents_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

    qa = RetrievalQA(combine_documents_chain=combine_documents_chain, retriever=retriever)

    user_input = st.text_input("Ask a question about your document:")

    if user_input:
        response = qa(user_input)["result"]
        st.write("**Response:**")
        st.write(response)
```

---

## Step 5: Running the App

Once your script is ready, start your Streamlit app:

```bash
streamlit run app.py
```

Open the link (usually `http://localhost:8501`) in your browser, upload a PDF, and ask questions about it. The system uses **DeepSeek R1** via **Ollama** for local inference, with **RAG** (Retrieval-Augmented Generation) powered by **LangChain**.

---

## Final Thoughts
- **✅ You have successfully set up Ollama and DeepSeek R1!**  
- **✅ You can now build AI-powered RAG applications with local LLMs!**  
- **✅ Try uploading PDFs and asking questions dynamically.**   

---

### Happy Building! 🚀

Feel free to open an issue or submit a pull request if you encounter any problems or have improvements to suggest. 

**Enjoy your local RAG system with DeepSeek R1!**


# Automation Plans


## 1. Overview of the Dockerized Approach

1. **Docker** creates a portable environment that includes:
   - Ollama (built from source or downloaded if binaries are available)
   - All Python dependencies (LangChain, Streamlit, etc.)
   - Your Streamlit app (`app.py`)
2. **Dockerfile** instructions:
   - Install system dependencies
   - Install Ollama
   - Pull the **DeepSeek R1** model
   - Install Python packages
   - Run the Streamlit app
3. **Single Command** to build and run the container. Once running, the app is available at `http://localhost:8501`.

> **Note**: Since Ollama is currently more straightforward to install on macOS than on Linux/Windows, the Docker route may require building from source on some platforms (especially Linux). Adjust the steps for your OS or reference Ollama’s official docs for the best approach. 

---

## 2. Create a Dockerfile

Create a file named `Dockerfile` in your project root (e.g., in the same folder as `app.py`):

```dockerfile
# Use Ubuntu as a base (or any Linux distribution you prefer)
FROM ubuntu:22.04

# Install core dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    python3 \
    python3-pip \
    cmake \
    clang \
    lld

# Clone and build Ollama from source
# (If there's a precompiled binary for your OS, you can skip building from source
# and just download + install that instead.)
RUN git clone https://github.com/jmorganca/ollama.git /ollama
WORKDIR /ollama
RUN make install  # Adjust if Ollama's build steps change

# Go back to root and clean up
WORKDIR /

# Upgrade pip and install Python packages
RUN pip3 install --upgrade pip
RUN pip3 install \
    streamlit \
    langchain \
    langchain-community \
    pdfplumber \
    semantic-chunkers \
    open-text-embeddings \
    faiss \
    ollama \
    prompt-template \
    langchain_experimental \
    sentence-transformers \
    faiss-cpu

# Copy your Streamlit app code into the container
COPY app.py /app/app.py
WORKDIR /app

# Expose Streamlit's default port
EXPOSE 8501

# Pull the DeepSeek R1 model before running the app
# Then launch Streamlit
CMD ollama pull deepseek-r1:1.5b && streamlit run app.py --server.address=0.0.0.0
```

### A Few Notes:
- **Make sure** your `app.py` references the correct imports and runs Streamlit without issues.  
- If the Ollama installation steps change, adjust the `Dockerfile` accordingly.  
- **Model storage**: The `ollama pull` command will download the model into the container. This increases the container size but ensures it’s pre-cached. Alternatively, you could skip pulling in `CMD` and do it interactively.

---

## 3. Build & Run the Docker Container

1. **Build the image** from your project directory:
   ```bash
   docker build -t deepseek-r1-rag .
   ```

2. **Run the container**, mapping port 8501 (Streamlit’s default):
   ```bash
   docker run -p 8501:8501 deepseek-r1-rag
   ```

3. **Access the app** by visiting:
   ```
   http://localhost:8501
   ```
   You’ll see your Streamlit interface. Once the container is up and running, you can upload a PDF and ask questions about it.  

---

## 4. (Optional) Docker Compose for Extended Flexibility

If you want to run multiple containers or add volumes for persistent data (e.g., storing PDFs, model files outside the container), create a `docker-compose.yml`:

```yaml
version: '3.8'
services:
  deepseek-r1-rag:
    build: .
    container_name: deepseek-r1-rag
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data  # Persist your PDF uploads, logs, etc.
    command: >
      /bin/bash -c "
      ollama pull deepseek-r1:1.5b &&
      streamlit run app.py --server.address=0.0.0.0
      "
```

Then run:

```bash
docker-compose up --build
```

This approach:
- **Builds** the Docker image.
- **Mounts** a local `./data` directory to `/app/data` in the container so data persists even if the container restarts.
- Automatically runs your Streamlit app.

---

## 5. Verifying Everything Works

1. **Open** `http://localhost:8501` after the container starts.  
2. **Upload a PDF** and submit a question.  
3. The system (DeepSeek R1 inside the Docker container) will:
   - Extract text from the PDF
   - Chunk and embed using LangChain
   - Retrieve relevant sections
   - Generate an answer with DeepSeek R1

---

## 6. Why This Is Cool

- **One-Step Deployment**: Anyone can clone your repo, run `docker build ...` and `docker run ...`, and get the exact same environment—no manual fuss with dependencies.  
- **Isolation**: Your local machine stays clean; everything runs inside Docker.  
- **Consistency**: Great for teams. Same container on a dev machine, on a server, or even in the cloud.  
- **Portability**: If you need to move from macOS to Linux or Windows, Docker smoothing out those platform differences is a huge plus.

---

### That’s It! 

With Docker (and optionally Docker Compose), you’ve now **automated** the entire DeepSeek R1 + Ollama + RAG + Streamlit pipeline. You can share this setup, spin up environments on new machines in minutes, and have confidence that it will “just work.”  

Enjoy your containerized local LLM adventures! If you have questions or hit any snags, feel free to tweak the Dockerfile, consult the [Ollama docs](https://ollama.ai/docs), or open a GitHub issue in your repository. 

**Happy automating!**