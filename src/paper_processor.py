import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from typing import List, Dict
import PyPDF2
import io

class PaperProcessor:
    def __init__(self):
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully!")
        
    def extract_paper_metadata(self, pdf_file) -> Dict:
        """Extract metadata from uploaded PDF file"""
        metadata = {
            'title': '',
            'authors': [],
            'abstract': '',
            'year': None,
            'doi': '',
            'references': [],
            'full_text': ''
        }
        
        try:
            # Handle both file path and file object
            if hasattr(pdf_file, 'read'):
                # It's a file object (from Streamlit upload)
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
            else:
                # It's a file path
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
            
            full_text = ""
            for page in pdf_reader.pages:
                try:
                    full_text += page.extract_text() + "\n"
                except:
                    continue
            
            metadata['full_text'] = full_text
            
            # Extract title (first meaningful line)
            lines = [line.strip() for line in full_text.split('\n') if line.strip()]
            if lines:
                # Usually title is one of the first few lines
                for line in lines[:10]:
                    if len(line) > 10 and not line.isupper():
                        metadata['title'] = line
                        break
            
            # Extract year (look for 4-digit year)
            year_match = re.search(r'\b(19|20)\d{2}\b', full_text)
            if year_match:
                metadata['year'] = int(year_match.group())
            
            # Extract references
            metadata['references'] = self.extract_citations(full_text)
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            metadata['title'] = "Error processing PDF"
            metadata['full_text'] = "Could not extract text from PDF"
        
        return metadata
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract citations from paper text"""
        citation_patterns = [
            r'\[(\d+(?:,\s*\d+)*)\]',  # [1], [1,2,3]
            r'\(([^)]+\d{4}[^)]*)\)',   # (Author 2023)
            r'(\w+\s+et\s+al\.\s+\d{4})'  # Smith et al. 2023
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        return list(set(citations))[:50]  # Limit to 50 citations