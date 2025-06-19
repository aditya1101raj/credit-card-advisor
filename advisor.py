__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from chromadb import PersistentClient 

# Setup model
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=pipe)

from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Model setup
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector store directory
persist_dir = "db"

# Build Chroma DB on first run
if not os.path.exists(persist_dir):
    from langchain.vectorstores import Chroma as ChromaSetup

    loader = JSONLoader(file_path="cards.json", jq_schema=".[]", text_content=False)More actions
    data = loader.load()
    print("[DEBUG] Loaded", len(data), "cards from JSON")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    for doc in data:
    doc.page_content = f"{doc.metadata.get('name', '')} {doc.metadata.get('perks', '')} {doc.metadata.get('fees', '')}"

    texts = splitter.split_documents(data)

    ChromaSetup.from_documents(
@@ -41,8 +46,12 @@
        persist_directory=persist_dir
    ).persist()

print("[DEBUG] Chroma DB created successfully.")

# Load DB (every time)
db = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
print("[DEBUG] Loaded DB with", len(db.get()['documents']), "documents")


# Conversation state
conversation_state = {
@@ -96,25 +105,26 @@
    )

    results = db.similarity_search(summary, k=3)
    print("[DEBUG] Querying with:", summary)

    if not results:
        return "Sorry, I couldn't find matching cards."

    response = " 💳 Here are some credit card recommendations based on your profile:\n"
    for i, doc in enumerate(results, 1):
        name = doc.metadata.get('name', 'Card')
        perks = doc.metadata.get('perks', 'No perks listed')
        fees = doc.metadata.get('fees', 'N/A')
        response += f"\n{i}. **{name}** — ₹{fees} annual fee — {perks}"
    return response
def reset_conversation():
    global step_index, session_started, conversation_state
    step_index = 0
    session_started = False
    conversation_state = {
        "income": None,
        "spending": None,
        "benefits": None,
        "existing_cards": None,
        "credit_score": None
    }
