import chromadb
from sentence_transformers import SentenceTransformer
import re
import numpy as np
from typing import List, Dict
from .citation_network import CitationNetworkBuilder

class AcademicRAG:
    def __init__(self):
        print("Initializing Academic RAG system...")
        self.client = chromadb.Client()
        self.collection = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.citation_network = CitationNetworkBuilder()
        print("Academic RAG system initialized!")
        
    def create_collection(self, collection_name: str = "academic_papers"):
        """Create or get ChromaDB collection"""
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {collection_name}")
    
    def chunk_academic_text(self, text: str, chunk_size: int = 512) -> List[str]:
        """Smart chunking for academic papers"""
        if not text or len(text.strip()) < 100:
            return [text] if text else []
        
        # Split by sections first (look for headers)
        sections = re.split(r'\n(?=[A-Z][A-Za-z\s]*:|\d+\.\s+[A-Z])', text)
        
        chunks = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            if len(section) <= chunk_size:
                chunks.append(section)
            else:
                # Split long sections by sentences
                sentences = re.split(r'(?<=[.!?])\s+', section)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) <= chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        # Filter out very short chunks
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def add_paper_to_index(self, paper_id: str, metadata: Dict):
        """Add paper to vector database and citation network"""
        try:
            # Add to citation network
            self.citation_network.add_paper(paper_id, metadata)
            
            # Chunk the text
            full_text = metadata.get('full_text', '')
            if not full_text:
                return False
            
            chunks = self.chunk_academic_text(full_text)
            
            if not chunks:
                return False
            
            # Create embeddings
            embeddings = self.embedding_model.encode(chunks).tolist()
            
            # Prepare documents with metadata
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{paper_id}_chunk_{i}"
                documents.append(chunk)
                metadatas.append({
                    'paper_id': paper_id,
                    'title': metadata.get('title', 'Unknown Title')[:100],
                    'authors': str(metadata.get('authors', []))[:200],
                    'year': metadata.get('year', 2023),
                    'chunk_index': i
                })
                ids.append(chunk_id)
            
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            return True
            
        except Exception as e:
            print(f"Error adding paper {paper_id}: {str(e)}")
            return False
    
    def retrieve_with_citation_context(self, query: str, n_results: int = 10) -> Dict:
        """Retrieve relevant papers with citation network context"""
        try:
            if not self.collection:
                return {'documents': [[]], 'metadatas': [[]], 'distances': [[]], 'citation_context': {}}
            
            # Get basic retrieval results
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count())
            )
            
            if not results['documents'][0]:
                return {'documents': [[]], 'metadatas': [[]], 'distances': [[]], 'citation_context': {}}
            
            # Enhance with citation network information
            enhanced_results = {
                'documents': results['documents'][0],
                'metadatas': results['metadatas'][0],
                'distances': results['distances'][0],
                'citation_context': {}
            }
            
            # Get influence metrics for retrieved papers
            influence_metrics = self.citation_network.calculate_influence_metrics()
            
            for metadata in results['metadatas'][0]:
                paper_id = metadata['paper_id']
                enhanced_results['citation_context'][paper_id] = {
                    'influence_score': influence_metrics.get(paper_id, {}).get('influence_score', 0.1),
                    'citation_count': influence_metrics.get(paper_id, {}).get('citations', 0),
                    'pagerank': influence_metrics.get(paper_id, {}).get('pagerank', 0)
                }
            
            return enhanced_results
            
        except Exception as e:
            print(f"Error in retrieval: {str(e)}")
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]], 'citation_context': {}}
    
    def generate_response(self, query: str, retrieved_docs: Dict) -> str:
        """Generate response using retrieved context"""
        try:
            if not retrieved_docs['documents']:
                return "I couldn't find any relevant papers for your query. Please try uploading some papers first or refining your search terms."
            
            # Sort documents by relevance and influence
            doc_scores = []
            for i, doc in enumerate(retrieved_docs['documents']):
                paper_id = retrieved_docs['metadatas'][i]['paper_id']
                retrieval_score = 1 - retrieved_docs['distances'][i]  # Convert distance to similarity
                influence_score = retrieved_docs['citation_context'][paper_id]['influence_score']
                
                combined_score = 0.7 * retrieval_score + 0.3 * influence_score
                doc_scores.append((i, combined_score))
            
            # Sort by combined score
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Create response
            response = f"Based on analysis of academic literature, here's what I found regarding: **{query}**\n\n"
            
            for i, (doc_idx, score) in enumerate(doc_scores[:5]):  # Top 5 results
                doc = retrieved_docs['documents'][doc_idx]
                metadata = retrieved_docs['metadatas'][doc_idx]
                paper_id = metadata['paper_id']
                citation_info = retrieved_docs['citation_context'][paper_id]
                
                response += f"**📄 Source {i+1}:** {metadata['title']}\n"
                response += f"*Year: {metadata['year']}, Citations: {citation_info['citation_count']}, Influence Score: {citation_info['influence_score']:.3f}*\n\n"
                
                # Truncate document content
                doc_preview = doc[:400] + "..." if len(doc) > 400 else doc
                response += f"{doc_preview}\n\n---\n\n"
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"Error generating response. Please try again or contact support."