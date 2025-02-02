import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader, S3FileLoader, AzureBlobStorageContainerLoader, GCSFileLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from smbclient import shutil, register_session
import pysftp

# ===== Style Configuration =====
primary_color = "#007BFF"
secondary_color = "#FFC107"
background_color = "#F8F9FA"
sidebar_background = "#2C2F33"
text_color = "#212529"
sidebar_text_color = "#FFFFFF"
header_text_color = "#000000"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {background_color}; color: {text_color}; }}
    [data-testid="stSidebar"] {{ 
        background-color: {sidebar_background} !important; 
        color: {sidebar_text_color} !important; 
    }}
    [data-testid="stSidebar"] * {{ color: {sidebar_text_color} !important; }}
    h1, h2, h3, h4, h5, h6 {{ color: {header_text_color} !important; font-weight: bold; }}
    p, span, div {{ color: {text_color} !important; }}
    .stFileUploader>div>div>div>button {{
        background-color: {secondary_color};
        color: #000000;
        font-weight: bold;
        border-radius: 8px;
    }}
    header {{ background-color: #1E1E1E !important; }}
    header * {{ color: #FFFFFF !important; }}
    </style>
""", unsafe_allow_html=True)

# ===== Sidebar Configuration =====
with st.sidebar:
    st.title("DeepSeek R1 RAG System")
    st.header("Instructions")
    st.markdown("""
    1. Choose document source type
    2. Load your document
    3. Ask questions about the content
    """)
    
    st.header("Storage Settings")
    storage_type = st.selectbox(
        "Document Source",
        ["Local Upload", "Windows Share (SMB)", "SFTP Server", "Cloud Storage"],
        index=0
    )

    # Storage-specific inputs
    remote_config = {}
    if storage_type != "Local Upload":
        if storage_type == "Windows Share (SMB)":
            remote_config.update({
                "server": st.text_input("SMB Server"),
                "share": st.text_input("Share Name"),
                "path": st.text_input("File Path"),
                "user": st.text_input("Username"),
                "password": st.text_input("Password", type="password")
            })
            
        elif storage_type == "SFTP Server":
            remote_config.update({
                "host": st.text_input("SFTP Host"),
                "path": st.text_input("File Path"),
                "user": st.text_input("Username"),
                "password": st.text_input("Password", type="password")
            })
            
        elif storage_type == "Cloud Storage":
            remote_config["provider"] = st.selectbox(
                "Cloud Provider",
                ["Azure Blob Storage", "AWS S3", "Google Cloud Storage"]
            )
            remote_config.update({
                "container": st.text_input("Container/Bucket Name"),
                "blob": st.text_input("Blob/File Name")
            })
            
            if remote_config["provider"] == "Azure Blob Storage":
                remote_config["conn_str"] = st.text_input("Connection String", type="password")
            elif remote_config["provider"] == "AWS S3":
                remote_config.update({
                    "aws_key": st.text_input("AWS Access Key", type="password"),
                    "aws_secret": st.text_input("AWS Secret Key", type="password")
                })
            elif remote_config["provider"] == "Google Cloud Storage":
                remote_config["gcp_creds"] = st.text_area("GCP Credentials JSON")

# ===== Main Application Logic =====
st.header("Document Processing")

# Initialize variables
docs = None
uploaded_file = None

if storage_type == "Local Upload":
    uploaded_file = st.file_uploader("Upload PDF file", type="pdf")
else:
    if st.button(f"Load from {storage_type}"):
        try:
            if storage_type == "Windows Share (SMB)":
                with st.spinner("Connecting to SMB share..."):
                    register_session(remote_config["server"], 
                                  username=remote_config["user"], 
                                  password=remote_config["password"])
                    local_path = f"/tmp/{os.path.basename(remote_config['path'])}"
                    shutil.copyfile(
                        f"\\\\{remote_config['server']}\\{remote_config['share']}\\{remote_config['path']}", 
                        local_path
                    )
                    uploaded_file = local_path
                    
            elif storage_type == "SFTP Server":
                with st.spinner("Connecting to SFTP server..."):
                    cnopts = pysftp.CnOpts()
                    # WARNING: Remove hostkeys bypass in production!
                    cnopts.hostkeys = None  
                    with pysftp.Connection(
                        remote_config["host"],
                        username=remote_config["user"],
                        password=remote_config["password"],
                        cnopts=cnopts
                    ) as sftp:
                        local_path = f"/tmp/{os.path.basename(remote_config['path'])}"
                        sftp.get(remote_config["path"], local_path)
                        uploaded_file = local_path
                        
            elif storage_type == "Cloud Storage":
                with st.spinner("Connecting to cloud storage..."):
                    provider = remote_config["provider"]
                    if provider == "Azure Blob Storage":
                        loader = AzureBlobStorageContainerLoader(
                            conn_str=remote_config["conn_str"],
                            container=remote_config["container"],
                            blob_name=remote_config["blob"]
                        )
                    elif provider == "AWS S3":
                        loader = S3FileLoader(
                            bucket=remote_config["container"],
                            key=remote_config["blob"],
                            aws_access_key_id=remote_config["aws_key"],
                            aws_secret_access_key=remote_config["aws_secret"]
                        )
                    elif provider == "Google Cloud Storage":
                        creds_path = "/tmp/gcp_creds.json"
                        with open(creds_path, "w") as f:
                            f.write(remote_config["gcp_creds"])
                        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
                        loader = GCSFileLoader(
                            project=remote_config["container"],
                            bucket=remote_config["container"],
                            blob=remote_config["blob"]
                        )
                    docs = loader.load()

            st.success("Document loaded successfully!")
            
        except Exception as e:
            st.error(f"Error loading document: {str(e)}")
            st.stop()

# Document processing pipeline
if uploaded_file or docs:
    try:
        # Handle file-based sources
        if uploaded_file:
            if storage_type in ["Windows Share (SMB)", "SFTP Server"]:
                loader = PDFPlumberLoader(uploaded_file)
                docs = loader.load()
                os.remove(uploaded_file)  # Cleanup temp file
            else:
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getvalue())
                loader = PDFPlumberLoader("temp.pdf")
                docs = loader.load()

        # Common processing for all document sources
        st.subheader("Processing Document")
        
        with st.spinner("Splitting document..."):
            text_splitter = SemanticChunker(HuggingFaceEmbeddings())
            documents = text_splitter.split_documents(docs)

        with st.spinner("Creating embeddings..."):
            embedder = HuggingFaceEmbeddings()
            vector = FAISS.from_documents(documents, embedder)
            retriever = vector.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 3}
            )

        # LLM Configuration
        llm = Ollama(model="deepseek-r1", base_url="http://localhost:11434")
        prompt_template = """
        1. Use context to answer the question
        2. If unsure, say "I don't know"
        3. Keep answers concise (3-4 sentences)
        Context: {context}
        Question: {question}
        Helpful Answer:"""
        
        QA_PROMPT = PromptTemplate.from_template(prompt_template)
        llm_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
        
        document_prompt = PromptTemplate(
            input_variables=["page_content", "source"],
            template="Content: {page_content}\nSource: {source}",
        )
        
        combine_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context",
            document_prompt=document_prompt
        )

        qa = RetrievalQA(
            combine_documents_chain=combine_chain,
            retriever=retriever,
            return_source_documents=True
        )

        # Question handling
        st.header("Ask Questions")
        user_input = st.text_input("Enter your question:")
        
        if user_input:
            with st.spinner("Analyzing document..."):
                try:
                    response = qa(user_input)["result"]
                    st.success("Answer:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")

    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        if "temp.pdf" in locals():
            os.remove("temp.pdf")
else:
    st.info("Please load a document to begin processing.")