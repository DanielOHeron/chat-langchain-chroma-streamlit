import re
import streamlit as st
from chain import answer_chain, format_chat_history

# Initialize session state variables if not already present
if 'history' not in st.session_state:
    st.session_state.history = []
    st.session_state.sources = []

# Function to update and display the conversation history and handle sources

# Define the text_input before defining the update_history function to avoid the error
user_input = st.text_input("Type your message here...", key="user_input")

# Function to update and display the conversation history and handle sources


def update_history():
    user_input = st.session_state.user_input
    if user_input:
        chat_history = format_chat_history(st.session_state.history)
        response = answer_chain.invoke(
            {"question": user_input, "chat_history": chat_history})

        # Update history with user input and AI response
        st.session_state.history.append(f"You: {user_input}")
        st.session_state.history.append(f"AI: {response}")

        # Parse the response for sources and format Python code
        response, new_sources = parse_response(
            response, st.session_state.sources)
        st.session_state.sources.extend(new_sources)

        # Clear input field
        st.session_state.user_input = ""
        return response


# Parse the response to extract sources and format Python code


def parse_response(response, sources):
    # Look for source references at the end of the response
    response_lines = response.strip().split('\n')
    last_line = response_lines[-1] if response_lines else ""
    source_ref_pattern = r"\[(\d+)\]"
    source_refs = re.findall(source_ref_pattern, last_line)

    if source_refs:
        # Remove the source references from the last line
        response_lines[-1] = re.sub(source_ref_pattern, '', last_line).strip()
        response = "\n".join(response_lines)

        # Convert source numbers to actual references
        new_sources = [
            f"Source {ref}: URL or document title here" for ref in source_refs]
    else:
        new_sources = []

    # Format Python code block if it exists
    response = format_python_code(response)
    return response, new_sources

# Format Python code block within the response


def format_python_code(response):
    python_code_pattern = r"```python(.*?)```"
    python_code_blocks = re.findall(python_code_pattern, response, re.DOTALL)
    formatted_response = response
    for code_block in python_code_blocks:
        # Replace the code block with a formatted Streamlit code block
        formatted_code_block = st.code(code_block.strip(), language='python')
        formatted_response = re.sub(
            python_code_pattern, formatted_code_block, formatted_response, count=1)
    return formatted_response


# Streamlit UI setup
st.title("LangChain Chat Application")
st.subheader("Ask a question and the AI will respond.")

# Send button
send_button = st.button("Send", on_click=update_history)
# Display conversation history and sources
st.subheader("Conversation History:")
history_container = st.container()
with history_container:
    for message in st.session_state.history:
        # Check for Python code block and format it
        if "```python" in message:
            st.code(message, language='python')
        else:
            st.text(message)

# Display sources
if st.session_state.sources:
    st.subheader("Sources:")
    for source in st.session_state.sources:
        st.text(source)

# Update the chat history after every input
if send_button or user_input:
    response = update_history(
        user_input, st.session_state.history, st.session_state.sources)
    history_container.text(response)
