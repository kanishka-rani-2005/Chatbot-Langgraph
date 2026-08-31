from agentic_chatbot_backend import chatbot
from langchain_core.messages import HumanMessage
import streamlit as st


st.title("Agentic Chatbot with LangGraph")

CONFIG = {
    "configurable": {
        "thread_id": "thread-1"
    }
}


# Initialize message history
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# Display previous conversation
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Get user input
user_input = st.chat_input("Type here")

if user_input:

    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):

        def generate_response():
            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages"
            ):
                content = message_chunk.content

                if isinstance(content, str):
                    yield content

                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            yield item["text"]

        ai_message = st.write_stream(generate_response())

    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
})