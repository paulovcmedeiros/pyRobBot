#!/usr/bin/env python3
# Adapted from
# <https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps>
import pickle
import sys

import streamlit as st

from chat_gpt.chat import Chat

# Initialize chat. Kepp it throughout the session.
try:
    session_chat = st.session_state["chat"]
except KeyError:
    parsed_args_file = sys.argv[-1]
    with open(parsed_args_file, "rb") as parsed_args_file:
        args = pickle.load(parsed_args_file)
    session_chat = Chat.from_cli_args(cli_args=args)
    st.session_state["chat"] = session_chat


page_title = f"Chat with {session_chat.model}"
# Set the title that is shown in the browser's tab
st.set_page_config(page_title=page_title)
# Set page title
st.title(page_title)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Send a message"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("▌")  # Use blinking cursor to indicate activity
        full_response = ""
        # Stream assistant response
        for chunk in session_chat.yield_response(prompt):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
