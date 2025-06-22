"""
Streamlit page for viewing retrieved chunks and retrieval details
"""
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Retrieved Chunks - RAG System",
    page_icon="📄", 
    layout="wide"
)

def main():
    st.title("📄 Retrieved Chunks & Retrieval Details")
    st.markdown("Hier sehen Sie die Details der letzten Dokumentensuche")
    
    # Check if there's a last result
    if 'last_result' not in st.session_state or 'last_query' not in st.session_state:
        st.info("🔍 Keine Suchergebnisse verfügbar. Bitte stellen Sie zuerst eine Frage auf der Hauptseite.")
        return
    
    result = st.session_state.last_result
    query = st.session_state.last_query
    
    # Header with query info
    st.header(f"🔍 Suchergebnisse für: '{query}'")
    
    # Query variants section
    st.subheader("🔄 Verwendete Suchanfragen")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Anzahl Anfrage-Varianten", len(result['query_variants']))
    
    with col2:
        for i, variant in enumerate(result['query_variants'], 1):
            variant_color = "🟢" if i == 1 else "🔵"
            label = "Original" if i == 1 else f"Variante {i-1}"
            st.write(f"{variant_color} **{label}:** {variant}")
    
    st.markdown("---")
    
    # Retrieval statistics
    if 'retrieval_details' in result:
        st.subheader("📊 Retrieval Statistiken")
        
        details = result['retrieval_details']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Gesamt gefundene Dokumente", details.get('total_unique_results', 0))
        
        with col2:
            st.metric("Suchanfrage-Varianten", details.get('total_variants', 0))
        
        with col3:
            successful_variants = sum(1 for v in details.get('results_per_variant', {}).values() 
                                    if 'error' not in v)
            st.metric("Erfolgreiche Varianten", successful_variants)
        
        # Detailed results per variant
        with st.expander("📈 Detaillierte Ergebnisse pro Suchanfrage"):
            for variant_key, variant_data in details.get('results_per_variant', {}).items():
                if 'error' in variant_data:
                    st.error(f"❌ {variant_data['query']}: {variant_data['error']}")
                else:
                    st.write(f"**{variant_data['query']}** → {variant_data['results_count']} Ergebnisse")
                    
                    if variant_data['results']:
                        df = pd.DataFrame(variant_data['results'])
                        st.dataframe(df, use_container_width=True)
    
    st.markdown("---")
    
    # Retrieved chunks section
    st.subheader("📝 Gefundene Dokumentenchunks")
    
    if not result['retrieved_chunks']:
        st.warning("Keine Chunks gefunden")
        return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Anzahl Chunks", len(result['retrieved_chunks']))
    
    with col2:
        companies = set(chunk['company'] for chunk in result['retrieved_chunks'])
        st.metric("Verschiedene Unternehmen", len(companies))
    
    with col3:
        avg_score = sum(chunk['score'] for chunk in result['retrieved_chunks']) / len(result['retrieved_chunks'])
        st.metric("Durchschnittlicher Score", f"{avg_score:.3f}")
    
    # Display chunks
    for i, chunk in enumerate(result['retrieved_chunks'], 1):
        with st.expander(f"📄 Chunk {i} - {chunk['company']} (Score: {chunk['score']:.3f})"):
            
            # Chunk metadata
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Unternehmen:** {chunk['company']}")
                st.write(f"**Quelle:** {chunk['source']}")
            
            with col2:
                st.write(f"**Relevanz-Score:** {chunk['score']:.3f}")
                st.write(f"**Gefunden mit Variante:** {chunk['variant_index'] + 1}")
            
            with col3:
                st.write(f"**Verwendete Anfrage:**")
                st.write(f"_{chunk['query_variant']}_")
            
            st.markdown("**Inhalt:**")
            st.text_area(
                "Chunk Inhalt:",
                value=chunk['content'],
                height=200,
                key=f"chunk_{i}",
                disabled=True
            )
    
    # Export functionality
    st.markdown("---")
    st.subheader("💾 Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Chunks als Text kopieren"):
            chunks_text = "\n\n" + "="*50 + "\n\n".join([
                f"CHUNK {i+1} - {chunk['company']} (Score: {chunk['score']:.3f})\n"
                f"Quelle: {chunk['source']}\n"
                f"Anfrage: {chunk['query_variant']}\n\n"
                f"{chunk['content']}"
                for i, chunk in enumerate(result['retrieved_chunks'])
            ])
            st.text_area("Kopieren Sie diesen Text:", value=chunks_text, height=200)
    
    with col2:
        # Create DataFrame for download
        chunks_df = pd.DataFrame([
            {
                'Chunk_Nr': i+1,
                'Unternehmen': chunk['company'],
                'Quelle': chunk['source'],
                'Score': chunk['score'],
                'Variante': chunk['variant_index'] + 1,
                'Suchanfrage': chunk['query_variant'],
                'Inhalt': chunk['content']
            }
            for i, chunk in enumerate(result['retrieved_chunks'])
        ])
        
        csv = chunks_df.to_csv(index=False)
        st.download_button(
            label="📊 Als CSV herunterladen",
            data=csv,
            file_name=f"retrieved_chunks_{query[:30]}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
