from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.schema.document import Document
import json

# File paths
CARD_JSON_PATH = "cards.json"        # Make sure this file exists in the same directory
CHROMA_DB_DIR = "db/"                # ChromaDB will save vectors here

# Load card data from JSON
with open(CARD_JSON_PATH, "r") as f:
    card_data = json.load(f)

# Convert to list of LangChain Documents
documents = []
for card in card_data:
    text = f"""
    Name: {card['name']}
    Issuer: {card['issuer']}
    Fee: {card['fees']}
    Rewards: {card['rewards']}
    Eligibility: {card['eligibility']}
    Perks: {card['perks']}
    Apply Link: {card['link']}
    """
    documents.append(Document(page_content=text.strip(), metadata=card))

# Create embeddings and store in Chroma
embedding = HuggingFaceEmbeddings()
db = Chroma.from_documents(documents, embedding, persist_directory=CHROMA_DB_DIR)
db.persist()

print(f"✅ Loaded {len(documents)} cards into ChromaDB at '{CHROMA_DB_DIR}'")
