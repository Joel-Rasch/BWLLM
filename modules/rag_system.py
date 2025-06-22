"""
RAG system that combines all modules
"""
from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

class RAGSystem:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
    
    def answer_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Answer question using RAG pipeline"""
        
        # Retrieve relevant documents
        retrieval_result = self.retriever.retrieve(question, top_k)
        documents = retrieval_result['documents']
        
        if not documents:
            return {
                'answer': 'Keine relevanten Informationen gefunden.',
                'sources': [],
                'query_variants': retrieval_result['query_variants'],
                'retrieval_details': retrieval_result['retrieval_details'],
                'retrieved_chunks': []
            }
        
        # Prepare context for answer generation
        context_parts = []
        retrieved_chunks = []
        
        for doc in documents[:8]:  # Limit context
            company = doc.metadata.get('company', 'Unknown')
            context_parts.append(f"[{company}]: {doc.page_content}")
            
            retrieved_chunks.append({
                'content': doc.page_content,
                'company': company,
                'source': doc.metadata.get('source', 'Unknown'),
                'score': doc.metadata.get('retrieval_score', 0),
                'query_variant': doc.metadata.get('query_variant', question),
                'variant_index': doc.metadata.get('variant_index', 0)
            })
        
        context = "\n\n".join(context_parts)
        
        # Create RAG prompt
        rag_prompt = ChatPromptTemplate.from_template(
            "Beantworte die Frage basierend auf dem Kontext aus deutschen Geschäftsberichten.\n\n"
            "Kontext:\n{context}\n\n"
            "Frage: {question}\n\n"
            "Antwort:"
        )
        
        # Generate answer
        try:
            chain = rag_prompt | self.llm | StrOutputParser()
            answer = chain.invoke({
                'context': context,
                'question': question
            })
            
            # Extract unique sources
            sources = list(set([doc.metadata.get('company', 'Unknown') for doc in documents[:5]]))
            
            return {
                'answer': answer.strip(),
                'sources': sources,
                'query_variants': retrieval_result['query_variants'],
                'retrieval_details': retrieval_result['retrieval_details'],
                'retrieved_chunks': retrieved_chunks
            }
            
        except Exception as e:
            return {
                'answer': f'Fehler bei der Antwortgenerierung: {str(e)}',
                'sources': [],
                'query_variants': retrieval_result['query_variants'],
                'retrieval_details': retrieval_result['retrieval_details'],
                'retrieved_chunks': retrieved_chunks
            }
