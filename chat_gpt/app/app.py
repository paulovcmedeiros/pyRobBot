import copy
import uuid

import page_template
import streamlit as st
from multipage import MultiPage

st.set_page_config(page_title="ChatGPT UI", page_icon=":speech_balloon:")

# Create an instance of the app
app = MultiPage()

available_chats = st.session_state.get("available_chats", {})

with st.sidebar:
    # Create a new chat upon init or button press
    if st.button(label="Create New Chat") or not available_chats:
        new_chat = {
            "page_id": str(uuid.uuid4()),
            "title": f"Chat {len(available_chats) + 1}",
            "func": copy.deepcopy(page_template.app),
        }
        app.add_page(**new_chat)
        available_chats[new_chat["page_id"]] = new_chat
        st.session_state["available_chats"] = available_chats

for chat in available_chats.values():
    app.add_page(**chat)

# Run the main app
app.run()
