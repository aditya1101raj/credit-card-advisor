import sys
import os
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma as ChromaSetup

# -------------------- Model Setup --------------------
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=pipe)

# -------------------- Embedding Setup --------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
persist_dir = "db"

# -------------------- Build Chroma DB if not exists --------------------
if not os.path.exists(persist_dir):
    loader = JSONLoader(file_path="cards.json", jq_schema=".[]", text_content=False)
    data = loader.load()
    st.write("[DEBUG] Loaded", len(data), "cards from JSON")

    for doc in data:
        doc.page_content = f"{doc.metadata.get('name', '')} {doc.metadata.get('perks', '')} {doc.metadata.get('fees', '')}"

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(data)

    ChromaSetup.from_documents(
        documents=texts,
        embedding=embedding_model,
        persist_directory=persist_dir
    ).persist()

    st.write("[DEBUG] Chroma DB created successfully.")

# -------------------- Load Chroma DB --------------------
db = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
st.write("[DEBUG] Loaded DB with", len(db.get()['documents']), "documents")

# -------------------- Conversation State --------------------
conversation_state = {
    "income": None,
    "spending": None,
    "benefits": None,
    "existing_cards": None,
    "credit_score": None
}

questions = [
    "What is your monthly income?",
    "Where do you spend the most? (fuel, groceries, travel, etc.)",
    "What kind of benefits do you prefer? (cashback, lounge access, travel points, etc.)",
    "Do you already have any credit cards?",
    "Do you know your credit score? You can say 'unknown' if you're not sure."
]

keys = list(conversation_state.keys())
step_index = 0
session_started = False

# -------------------- Main Chat Function --------------------
def get_response(user_input):
    global step_index, session_started

    cleaned_input = user_input.strip().lower()

    if not session_started:
        if cleaned_input not in ["hi", "hello", "start"]:
            return "Please type 'Hi' or 'Start' to begin the credit card recommendation process."
        session_started = True
        return "Welcome! Let's get started. " + questions[step_index]

    # Save user's input
    if step_index < len(keys):
        conversation_state[keys[step_index]] = user_input
        step_index += 1

    # Ask next question
    if step_index < len(questions):
        return questions[step_index]

    # All questions answered, generate summary
    summary = (
        f"My monthly income is {conversation_state['income']} rupees. "
        f"I mostly spend on {conversation_state['spending'].lower()}. "
        f"I'm looking for credit cards that offer {conversation_state['benefits'].lower()}. "
        f"I already have these cards: {conversation_state['existing_cards'].lower()}. "
        f"My credit score is {conversation_state['credit_score']}."
    )

    st.write("[DEBUG] Querying with:", summary)

    results = db.max_marginal_relevance_search(summary, k=3)
   
