# main.py
import os
import shutil
import json
from pathlib import Path
from typing import Dict, Optional, List
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import CORSMiddleware

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# NEW: Import ChatGoogleGenerativeAI for integration with Google Gemini LLMs
from langchain_google_genai import ChatGoogleGenerativeAI # Make sure you have installed this: pip install langchain-google-genai

# REMOVED: Import for ChatOpenAI as it's no longer needed
# from langchain_openai import ChatOpenAI

from langchain.retrievers import MergerRetriever

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Document RAG API",
    description="API for uploading documents and engaging in conversational RAG.",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
from fastapi.responses import HTMLResponse
@app.get("/", response_class=HTMLResponse)
async def read_root_frontend():
    with open(Path("static/index.html"), "r") as f:
        return f.read()
origins = [
    "http://localhost",
    "http://localhost:8001",
    # Add your deployed frontend URL here later, e.g.:
    # "https://your-deployed-frontend.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIRECTORY = Path("uploaded_documents")
UPLOAD_DIRECTORY.mkdir(exist_ok=True)

CHROMA_DB_DIRECTORY = Path("chroma_db")
CHROMA_DB_DIRECTORY.mkdir(exist_ok=True)

DOCUMENTS_METADATA: Dict[str, Dict] = {}

print("Loading embedding model...")
try:
    # Note: For Google models, you might consider GoogleGenerativeAIEmbeddings
    # but all-MiniLM-L6-v2 is a good general-purpose embedding model.
    EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Embedding model loaded successfully.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    EMBEDDING_MODEL = None

LLM_MODEL = None
print("Loading LLM model (Google Gemini)...")
try:
    # Initialize ChatGoogleGenerativeAI with API key from environment variables
    # The GOOGLE_API_KEY will be automatically picked up by LangChain if it's in your .env
    # Using 'gemini-pro' for text-based chat. For multimodal, you might use 'gemini-pro-vision'.
    LLM_MODEL = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    print("LLM model (Google Gemini - gemini-2.0-flash	) loaded successfully.")
except Exception as e:
    print(f"Error loading LLM model (Google Gemini): {e}")
    print("Please ensure your GOOGLE_API_KEY is set in your .env file and is valid.")
    LLM_MODEL = None

def get_chroma_collection_name(document_id: str) -> str:
    return f"doc-{document_id}"

def save_documents_metadata():
    meta_file = Path("documents_metadata.json")
    try:
        with open(meta_file, "w") as f:
            json.dump(DOCUMENTS_METADATA, f, indent=4)
        print("Documents metadata saved.")
    except Exception as e:
        print(f"Error saving documents metadata: {e}")

async def process_document_background(document_id: str, file_path: Path, file_extension: str):
    print(f"Starting background processing for document_id: {document_id}")
    print(f"File path: {file_path}, Extension: {file_extension}")

    if EMBEDDING_MODEL is None:
        print(f"Skipping processing for {document_id}: Embedding model not loaded.")
        DOCUMENTS_METADATA[document_id]["status"] = "failed"
        save_documents_metadata()
        return

    try:
        loader = None
        if file_extension == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_extension == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif file_extension == ".docx":
            loader = Docx2txtLoader(str(file_path))
        else:
            raise ValueError(f"Unsupported file extension for processing: {file_extension}")

        loaded_documents = loader.load()
        print(f"Loaded {len(loaded_documents)} parts from {file_path.name}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(loaded_documents)
        print(f"Split document into {len(chunks)} chunks.")

        for i, chunk in enumerate(chunks):
            chunk.metadata["source_document_id"] = document_id
            chunk.metadata["original_filename"] = DOCUMENTS_METADATA[document_id]["filename"]
            chunk.metadata["chunk_index"] = i
            if "page" in chunk.metadata:
                chunk.metadata["page_number"] = chunk.metadata["page"]
                del chunk.metadata["page"]

        collection_name = get_chroma_collection_name(document_id)

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=EMBEDDING_MODEL,
            collection_name=collection_name,
            persist_directory=str(CHROMA_DB_DIRECTORY)
        )
        vector_store.persist()
        print(f"Chunks and embeddings stored in ChromaDB collection '{collection_name}'.")

        DOCUMENTS_METADATA[document_id]["status"] = "ready"
        print(f"Document {document_id} processed successfully and ready for querying.")
        save_documents_metadata()

    except ValueError as ve:
        print(f"Processing error for document {document_id}: {ve}")
        DOCUMENTS_METADATA[document_id]["status"] = "failed"
        save_documents_metadata()
    except Exception as e:
        print(f"Unexpected error processing document {document_id}: {e}")
        DOCUMENTS_METADATA[document_id]["status"] = "failed"
        save_documents_metadata()
    finally:
        pass


@app.post("/api/documents/upload", status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded or filename is empty.")

    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in [".pdf", ".txt", ".docx"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only .pdf, .txt, .docx are allowed."
        )

    document_id = str(uuid.uuid4())
    temp_upload_dir = UPLOAD_DIRECTORY / document_id
    temp_upload_dir.mkdir(exist_ok=True)
    file_location = temp_upload_dir / f"{file.filename}"

    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"File '{file.filename}' saved to '{file_location}'")

        DOCUMENTS_METADATA[document_id] = {
            "id": document_id,
            "filename": file.filename,
            "file_path": str(file_location),
            "upload_timestamp": os.path.getctime(file_location),
            "status": "processing",
            "chroma_collection_name": get_chroma_collection_name(document_id)
        }

        background_tasks.add_task(
            process_document_background,
            document_id,
            file_location,
            file_extension
        )

        return JSONResponse(content={
            "message": "Document upload initiated successfully.",
            "document_id": document_id,
            "filename": file.filename,
            "status": "processing"
        })
    except Exception as e:
        print(f"Error during file upload: {e}")
        raise HTTPException(status_code=500, detail=f"Could not upload file: {e}")

@app.get("/api/documents/{document_id}/status")
async def get_document_status(document_id: str):
    doc_info = DOCUMENTS_METADATA.get(document_id)
    if not doc_info:
        raise HTTPException(status_code=404, detail="Document not found.")

    return JSONResponse(content={
        "document_id": document_id,
        "filename": doc_info["filename"],
        "status": doc_info["status"]
    })

@app.get("/api/documents")
async def list_documents():
    documents_list = [
        {"id": doc["id"], "filename": doc["filename"], "status": doc["status"]}
        for doc_id, doc in DOCUMENTS_METADATA.items()
    ]
    return JSONResponse(content={"documents": documents_list})

class ChatQuery(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None

QA_PROMPT_TEMPLATE = """
You are a helpful and highly accurate AI assistant. Your goal is to provide concise, direct answers to the user's question, based solely on the provided context.
If the information to answer the question is not present in the context, state clearly that you do not have enough information to answer the question from the provided documents.
Do not invent or extrapolate information.

Context:
{context}

Question: {question}

Concise Answer:
"""
CUSTOM_QA_PROMPT = PromptTemplate(template=QA_PROMPT_TEMPLATE, input_variables=["context", "question"])


@app.post("/api/chat")
async def chat_with_document(chat_query: ChatQuery):
    query = chat_query.query
    document_ids = chat_query.document_ids

    if LLM_MODEL is None or EMBEDDING_MODEL is None:
        raise HTTPException(status_code=500, detail="LLM or Embedding model not loaded. Cannot process query.")

    try:
        retrievers = []
        if document_ids:
            for doc_id in document_ids:
                doc_info = DOCUMENTS_METADATA.get(doc_id)
                if not doc_info or doc_info["status"] != "ready":
                    raise HTTPException(
                        status_code=400,
                        detail=f"Selected document '{doc_info.get('filename', doc_id)}' is not found or not ready for querying."
                    )
                collection_name = get_chroma_collection_name(doc_id)
                retrievers.append(
                    Chroma(
                        collection_name=collection_name,
                        embedding_function=EMBEDDING_MODEL,
                        persist_directory=str(CHROMA_DB_DIRECTORY)
                    ).as_retriever(search_kwargs={"k": 10})
                )
            print(f"Querying across selected documents: {document_ids}")

            if not retrievers:
                raise HTTPException(status_code=400, detail="No ready documents selected for chat.")

            combined_retriever = MergerRetriever(retrievers=retrievers)

        else:
            raise HTTPException(status_code=400, detail="Please select at least one document to chat with.")


        qa_chain = RetrievalQA.from_chain_type(
            llm=LLM_MODEL,
            chain_type="stuff",
            retriever=combined_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": CUSTOM_QA_PROMPT},
        )

        print(f"Received query: {query}")
        result = qa_chain.invoke({"query": query})

        response_text = result.get("result", "No relevant answer found from the provided documents.")
        source_documents = result.get("source_documents", [])

        # This block is commented out to remove the "Sources:" section
        # sources_info = []
        # seen_sources = set()
        # for doc in source_documents:
        #     filename = doc.metadata.get("original_filename", "Unknown File")
        #     page_info = f"Page {doc.metadata.get('page_number')}" if doc.metadata.get('page_number') else ""
        #     chunk_info = f"Chunk {doc.metadata.get('chunk_index')}" if doc.metadata.get('chunk_index') else ""

        #     source_identifier = (filename, page_info, chunk_info)
        #     if source_identifier not in seen_sources:
        #         sources_info.append(f"- {filename} ({page_info} {chunk_info})")
        #         seen_sources.add(source_identifier)

        # if sources_info:
        #     response_text += "\n\n**Sources:**\n" + "\n".join(sources_info)

        return JSONResponse(content={"response": response_text})

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error during chat query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during chat: {e}")


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Document RAG API!"}

@app.on_event("startup")
async def startup_event():
    print("Loading existing ChromaDB collections on startup...")
    try:
        meta_file = Path("documents_metadata.json")
        if meta_file.exists():
            with open(meta_file, "r") as f:
                loaded_meta = json.load(f)
                for doc_id, doc_data in loaded_meta.items():
                    DOCUMENTS_METADATA[doc_id] = doc_data
                    print(f"Loaded document metadata for {doc_data['filename']} (Status: {doc_data['status']})")
        else:
            print("No documents_metadata.json found. Starting with empty document list.")
    except Exception as e:
        print(f"Error loading existing document metadata: {e}")
