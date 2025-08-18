import streamlit as st
from rag_system import query_faiss_index, rag, process_query
import variable_loader as loader
loader.load_variables()

def get_rag_response(query):
    # Top-k Dokumente aus dem Kontext abrufen
    matched_companies, query_cleaned = process_query(query)
    context = query_faiss_index(query_cleaned, matched_companies)
    # Antwort mittels RAG-Pipeline generieren
    answer = rag(question=query).content
    return answer, context


def rag_chatbot():
    st.title("RAG Chatbot")

    # Chat-Verlauf initialisieren
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat-Verlauf anzeigen
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Benutzereingabe
    if prompt := st.chat_input("Was möchten Sie wissen?"):
        # Nachricht zum Chat-Verlauf hinzufügen
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Benutzernachricht anzeigen
        with st.chat_message("user"):
            st.markdown(prompt)

        # Antwort und Kontext von RAG abrufen
        response, context = get_rag_response(prompt)

        # Assistenten-Antwort anzeigen
        with st.chat_message("assistant"):
            st.markdown(response)
            # Ausklappbaren Kontextbereich hinzufügen
            with st.expander("🔍 Kontext Anschauen"):
                st.markdown("### Kontext")
                for i, doc in enumerate(context, start=1):
                    st.markdown(f"**{i}. Dokument**")
                    st.markdown(f"- **ID:** `{doc.id}`")
                    st.markdown(f"- **Metadata:** `{doc.metadata}`")
                    st.markdown(f"- **Content:** {doc.page_content}")
                    st.markdown("---")

        # Assistenten-Antwort zum Chat-Verlauf hinzufügen (ohne Kontext)
        st.session_state.messages.append({"role": "assistant", "content": response})


def dummy_page():
    st.title("Testseite")
    query = st.text_input("Geben Sie Ihre Suchanfrage ein:", "")
    if query:
        matched_companies, query_cleaned = process_query(query)
        results = query_faiss_index(query_cleaned, matched_companies)
        st.markdown("### Ergebnisse:")
        st.markdown(f"**Ergebnis:** {results}")


# Seitennavigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seite auswählen", ["RAG Chatbot", "Testseite"])

if page == "RAG Chatbot":
    rag_chatbot()
elif page == "Testseite":
    dummy_page()
