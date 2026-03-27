#pip install langchain openai pinecone-client langchain-community langchain-openai

# ============================================
# IMPORTS
# ============================================
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import RetrievalQA
import hashlib
import os
import json

# ============================================
# ENVIRONMENT VARIABLES
# ============================================
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
PINECONE_API_KEY = "YOUR_PINECONE_API_KEY"

# ============================================
# PERSISTENT CACHE (Simple JSON-Based)
# ============================================
CACHE_FILE = "longterm_cache.json"

def load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}
    with open(CACHE_FILE, "r") as f:
        return json.load(f)

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)

cache = load_cache()


# ============================================
# CACHE HANDLER
# ============================================
def cache_key(user_id, question):
    raw = f"{user_id}:{question}"
    return hashlib.sha256(raw.encode()).hexdigest()

def check_cache(user_id, question):
    key = cache_key(user_id, question)
    return cache.get(key, None)

def store_cache(user_id, question, answer):
    key = cache_key(user_id, question)
    cache[key] = answer
    save_cache(cache)


# ============================================
# PINECONE VECTOR DB INITIALIZATION
# ============================================
pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "longterm-memory"

# Create index if NOT exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# LangChain Wrappers
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(index, embeddings)


# ============================================
# LONG-TERM MEMORY RETRIEVER (Pinecone)
# ============================================
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ============================================
# LLM MODEL
# ============================================
llm = ChatOpenAI(model="gpt-4.1", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)


# ============================================
# MAIN QA FUNCTION
# ============================================
def ask_ai(user_id, question):
    # ✅ 1. Check session-based long-term cache
    cached_answer = check_cache(user_id, question)
    if cached_answer:
        print("\n[✅ Retrieved from Persistent Cache]\n")
        return cached_answer

    # ✅ 2. Query long-term memory (Pinecone)
    print("\n[🔍 Searching Pinecone Long-Term Memory...]\n")
    memory_answer = qa_chain.run(question)

    # ✅ 3. Store new memory into Pinecone (embedding)
    vectorstore.add_texts([question + " -> " + memory_answer])

    # ✅ 4. Save to cache
    store_cache(user_id, question, memory_answer)

    return memory_answer


# ============================================
# EXAMPLE USAGE
# ============================================
if __name__ == "__main__":
    user_id = "user_123"
    while True:
        q = input("\nAsk: ")
        if q.lower() == "exit":
            break
        answer = ask_ai(user_id, q)
        print("\nAI:", answer)
