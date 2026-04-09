# 🔍 Semantic Search App

A full-stack AI project using **Transformers + Vector Database + FastAPI + Streamlit**.

## 🏗️ Architecture
```
Streamlit UI → FastAPI Backend → SentenceTransformers → ChromaDB Vector DB
```

## 🤖 Tech Stack
| Technology | Purpose |
|---|---|
| `sentence-transformers` | Convert text to vectors (embeddings) |
| `ChromaDB` | Vector database - store & search embeddings |
| `FastAPI` | REST API backend |
| `Streamlit` | Frontend UI |

## 💡 How It Works
1. Text is converted to a **vector** (list of numbers) using Transformers
2. Vectors are stored in **ChromaDB** vector database
3. When you search, your query is also converted to a vector
4. ChromaDB finds documents with **similar vectors** = similar meaning
5. Results ranked by **cosine similarity score**

## 🚀 How to Run

### Step 1 - Install
```bash
pip install -r requirements.txt
```

### Step 2 - Start FastAPI backend
```bash
python main.py
```
✅ API: http://localhost:8000
✅ API Docs: http://localhost:8000/docs

### Step 3 - Start Streamlit (new terminal)
```bash
streamlit run streamlit_app.py
```

## 📡 API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | / | API info |
| GET | /health | Health check |
| POST | /add | Add document to vector DB |
| POST | /search | Semantic search |
| GET | /documents | List all documents |
| DELETE | /clear | Clear database |

## 📁 Project Structure
```
semantic-search/
├── main.py            ← FastAPI + Transformers + ChromaDB
├── streamlit_app.py   ← Streamlit frontend
├── requirements.txt
└── README.md
```
