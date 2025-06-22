"""
Query enhancement module for improving retrieval
"""
from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

class QueryEnhancer:
    def __init__(self, llm):
        self.llm = llm
    
    def enhance_query(self, query: str, max_variants: int = 4) -> List[str]:
        """Generate multiple query variants for better retrieval"""
        prompt = ChatPromptTemplate.from_template(
            "Erstelle {max_variants} alternative Suchbegriffe für diese Frage in deutschen Geschäftsberichten:\n"
            "'{query}'\n\n"
            "Gib nur die {max_variants} Alternativen zurück, eine pro Zeile, ohne Nummerierung:"
        )
        
        try:
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "query": query,
                "max_variants": max_variants - 1  # -1 because we include original
            })
            
            alternatives = [line.strip() for line in response.split('\n') if line.strip()]
            
            # Ensure we don't exceed max_variants
            alternatives = alternatives[:max_variants-1]
            
            # Return original query + alternatives
            return [query] + alternatives
            
        except Exception as e:
            # Fallback: return original query with basic variations
            return [
                query,
                f"Kennzahlen {query}",
                f"Finanzielle Entwicklung {query}",
                f"Geschäftsbericht {query}"
            ][:max_variants]
