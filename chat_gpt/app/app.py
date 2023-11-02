import copy
import uuid

import page_template
import streamlit as st
from multipage import MultiPage

# Create an instance of the app
app = MultiPage()

# Title of the main page
st.title("Chat GPT UI")

available_chats = st.session_state.get("available_chats", [])

with st.sidebar:
    if st.button(label="Create New Chat"):
        # Add all your applications (pages) here
        new_chat = {
            "page_id": str(uuid.uuid4()),
            "title": f"Chat {len(available_chats) + 1}",
            "func": copy.deepcopy(page_template.app),
        }
        app.add_page(**new_chat)
        available_chats.append(new_chat)
        st.session_state["available_chats"] = available_chats

for chat in available_chats:
    app.add_page(**chat)


# The main app
app.run()
