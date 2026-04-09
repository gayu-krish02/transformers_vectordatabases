from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uvicorn
import uuid

# ============================================================
# Semantic Search API
# Transformers + ChromaDB (Vector DB) + FastAPI
# ============================================================

app = FastAPI(
    title="Semantic Search API",
    description="Store and search documents using vector embeddings",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load Transformer Model ----
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded!")

# ---- Setup ChromaDB (Vector Database) ----
print("Setting up ChromaDB...")
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)
print("✅ ChromaDB ready!")

# ---- Add sample documents on startup ----
sample_docs = [
    "Python is a high-level programming language known for simplicity.",
    "Machine learning is a subset of artificial intelligence.",
    "FastAPI is a modern web framework for building APIs with Python.",
    "Vector databases store data as mathematical vectors for similarity search.",
    "Transformers are deep learning models used in NLP tasks.",
    "ChromaDB is an open-source vector database for AI applications.",
    "Streamlit is used to build interactive web apps with Python.",
    "Neural networks are inspired by the human brain structure.",
    "Natural language processing helps computers understand human language.",
    "Deep learning uses multiple layers of neural networks.",
]

if collection.count() == 0:
    print("Adding sample documents...")
    embeddings = embedding_model.encode(sample_docs).tolist()
    collection.add(
        documents=sample_docs,
        embeddings=embeddings,
        ids=[str(uuid.uuid4()) for _ in sample_docs]
    )
    print(f"✅ Added {len(sample_docs)} sample documents!")


# ---- Request / Response Models ----
class DocumentRequest(BaseModel):
    text: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class SearchResult(BaseModel):
    document: str
    similarity: float
    rank: int


# ---- API Routes ----

@app.get("/")
def root():
    return {
        "message": "Semantic Search API is running!",
        "total_documents": collection.count(),
        "endpoints": {
            "add_document": "POST /add",
            "search": "POST /search",
            "list_all": "GET /documents",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "documents_in_db": collection.count()
    }

@app.post("/add")
def add_document(request: DocumentRequest):
    """Add a new document to the vector database."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        embedding = embedding_model.encode([request.text]).tolist()
        doc_id = str(uuid.uuid4())
        collection.add(
            documents=[request.text],
            embeddings=embedding,
            ids=[doc_id]
        )
        return {
            "message": "Document added successfully!",
            "id": doc_id,
            "total_documents": collection.count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search_documents(request: SearchRequest):
    """Search for similar documents using semantic similarity."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if collection.count() == 0:
        raise HTTPException(status_code=404, detail="No documents in database")

    try:
        query_embedding = embedding_model.encode([request.query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(request.top_k, collection.count())
        )

        search_results = []
        documents = results["documents"][0]
        distances = results["distances"][0]

        for i, (doc, dist) in enumerate(zip(documents, distances)):
            similarity = round((1 - dist) * 100, 2)
            search_results.append(SearchResult(
                document=doc,
                similarity=similarity,
                rank=i + 1
            ))

        return {
            "query": request.query,
            "results": search_results,
            "total_found": len(search_results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def list_documents():
    """List all documents in the vector database."""
    if collection.count() == 0:
        return {"documents": [], "total": 0}

    try:
        results = collection.get()
        return {
            "documents": results["documents"],
            "total": len(results["documents"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear")
def clear_database():
    """Clear all documents from the vector database."""
    global collection
    chroma_client.delete_collection("documents")
    collection = chroma_client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )
    return {"message": "Database cleared!", "total_documents": 0}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
