import streamlit as st
import rag_system

def get_rag_response(query):
    # This is a placeholder function that will be replaced with actual RAG implementation
    answer = rag_system.rag(query)
    return answer

def main():
    st.title("RAG Chatbot")
    st.write("Ask me anything! (Currently using placeholder responses)")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from RAG (currently placeholder)
        response = get_rag_response(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()