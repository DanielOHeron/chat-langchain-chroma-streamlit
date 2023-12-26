import streamlit as st
from chain import answer_chain

# Initialize session state for conversation history if not already present
if 'history' not in st.session_state:
    st.session_state.history = []

# Function to update and display the conversation history


def update_history(user_input):
    if user_input:
        # Append user input to history
        st.session_state.history.append(f"You: {user_input}")

        # Generate a response using the answer_chain
        response = answer_chain.invoke(
            {"question": user_input, "chat_history": []})

        # Append AI response to history
        st.session_state.history.append(f"AI: {response}")


# Streamlit UI setup
st.title("Chat Application with AI")
st.subheader("Type your message and get an AI response.")

# Text input for user message
user_input = st.text_input("Type your message here...", key="user_input")

# Send button
if st.button("Send"):
    update_history(user_input)

# Display conversation history
st.subheader("Conversation History:")
for message in st.session_state.history:
    #
    #     st.text(message)
    st.markdown(
        f"<p style='word-wrap: break-word;'>{message}</p>", unsafe_allow_html=True)
