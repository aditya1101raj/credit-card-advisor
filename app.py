import streamlit as st
from advisor import get_response


import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # optional warning suppression

# Patch sqlite3 to use pysqlite3 (which is >= 3.35)
import pysqlite3
sys.modules["sqlite3"] = pysqlite3


st.set_page_config(page_title="Credit Card Advisor", layout="centered")
st.title("💳 AI Credit Card Advisor")

# --- CSS for WhatsApp-like chat ---
st.markdown("""
    <style>
    .user-bubble {
        background-color: #25D366;
        color: white;
        padding: 10px 15px;
        border-radius: 20px;
        max-width: 70%;
        margin: 5px 0 5px auto;
        text-align: left;
        font-size: 16px;
    }
    .bot-bubble {
        background-color: #f1f0f0;
        color: black;
        padding: 10px 15px;
        border-radius: 20px;
        max-width: 70%;
        margin: 5px auto 5px 0;
        text-align: left;
        font-size: 16px;
    }
.chat-container {
    max-height: 500px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
}


    </style>
""", unsafe_allow_html=True)

# --- Chat state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Welcome message ---
if not st.session_state.chat_history:
    st.markdown("""
    <div class="bot-bubble">
        👋 Welcome! I'm your AI Credit Card Advisor.<br>
        Type <b>Hi</b> to begin, and I'll ask you a few simple questions to find the best cards for you.
    </div>
    """, unsafe_allow_html=True)

# --- Chat Display ---
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for user_msg, bot_msg in st.session_state.chat_history:

    st.markdown(f'<div class="user-bubble">{user_msg}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bot-bubble">{bot_msg}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Input ---
query = st.chat_input("Type your message here...")
if query:
    response = get_response(query)
    st.session_state.chat_history.append((query, response))
    st.rerun()

# --- Restart Button ---
if st.button("🔄 Restart Chat"):
    from advisor import reset_conversation
    reset_conversation()
    st.session_state.chat_history = []
    st.rerun()

