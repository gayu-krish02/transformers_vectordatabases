import streamlit as st
import requests

# ============================================================
# Streamlit Frontend - Semantic Search App
# ============================================================

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Semantic Search",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Semantic Search App")
st.caption("Powered by Transformers + ChromaDB Vector Database + FastAPI")

# ---- Check API health ----
try:
    res = requests.get(f"{API_URL}/health", timeout=3)
    data = res.json()
    st.success(f"✅ API Running! Documents in DB: {data['documents_in_db']}")
except:
    st.error("❌ API is offline! Run: `python main.py` in a terminal first.")
    st.stop()

st.markdown("---")

# ---- TABS ----
tab1, tab2, tab3 = st.tabs(["🔍 Search", "➕ Add Document", "📄 View All Documents"])

# ---- SEARCH TAB ----
with tab1:
    st.subheader("🔍 Semantic Search")
    st.write("Search documents by **meaning**, not just keywords!")

    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g. What is deep learning?"
    )

    top_k = st.slider("Number of results:", min_value=1, max_value=10, value=3)

    if st.button("🔍 Search", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching..."):
                try:
                    response = requests.post(
                        f"{API_URL}/search",
                        json={"query": query, "top_k": top_k},
                        timeout=30
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Found {data['total_found']} results!")

                        for result in data["results"]:
                            with st.container():
                                col1, col2 = st.columns([4, 1])
                                col1.write(f"**#{result['rank']}** {result['document']}")
                                similarity = result['similarity']
                                color = "🟢" if similarity > 70 else "🟡" if similarity > 40 else "🔴"
                                col2.metric("Similarity", f"{color} {similarity}%")
                                st.divider()
                    else:
                        st.error(f"Error: {response.json()['detail']}")
                except Exception as e:
                    st.error(f"Error: {e}")

# ---- ADD DOCUMENT TAB ----
with tab2:
    st.subheader("➕ Add Document to Vector DB")
    st.write("Add your own documents to be searchable!")

    new_doc = st.text_area(
        "Enter document text:",
        placeholder="e.g. Streamlit makes it easy to build data apps in Python.",
        height=150
    )

    if st.button("➕ Add to Database", use_container_width=True):
        if not new_doc.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Adding to vector database..."):
                try:
                    response = requests.post(
                        f"{API_URL}/add",
                        json={"text": new_doc},
                        timeout=30
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Document added! Total documents: {data['total_documents']}")
                        st.info(f"Document ID: `{data['id']}`")
                    else:
                        st.error(f"Error: {response.json()['detail']}")
                except Exception as e:
                    st.error(f"Error: {e}")

# ---- VIEW ALL TAB ----
with tab3:
    st.subheader("📄 All Documents in Vector DB")

    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

    try:
        response = requests.get(f"{API_URL}/documents", timeout=10)
        if response.status_code == 200:
            data = response.json()
            st.info(f"Total documents: **{data['total']}**")

            for i, doc in enumerate(data["documents"], 1):
                st.write(f"**{i}.** {doc}")
                st.divider()

            if st.button("🗑️ Clear All Documents", type="secondary"):
                clear_res = requests.delete(f"{API_URL}/clear", timeout=10)
                if clear_res.status_code == 200:
                    st.success("✅ Database cleared!")
                    st.rerun()
    except Exception as e:
        st.error(f"Error: {e}")

# ---- FOOTER ----
st.markdown("---")
col1, col2 = st.columns(2)
col1.markdown("**How it works:**")
col1.markdown("""
1. Documents → converted to vectors using **Transformers**
2. Vectors stored in **ChromaDB** vector database
3. Search query → converted to vector → find similar vectors
4. Results ranked by **cosine similarity**
""")
col2.markdown("**API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)")
col2.markdown("**Tech Stack:** `sentence-transformers` + `chromadb` + `fastapi` + `streamlit`")
