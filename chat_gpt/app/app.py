# Adapted from
# <https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps>
import streamlit as st

from chat_gpt.chat_gpt import Chat

# Initialize chat. Kepp it throughout the session.
try:
    session_chat = st.session_state["chat"]
except KeyError:
    session_chat = Chat(
        model="gpt-3.5-turbo",
        base_instructions="You answer using the minimum possible number of tokens.",
    )
    st.session_state["chat"] = session_chat

st.title(f"Chat with {session_chat.assistant_name}")

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
