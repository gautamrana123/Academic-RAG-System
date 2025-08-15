import networkx as nx
from datetime import datetime
import re
from typing import Dict, List

class CitationNetworkBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.paper_embeddings = {}
    
    def add_paper(self, paper_id: str, metadata: Dict):
        """Add paper to citation network"""
        self.graph.add_node(paper_id, **metadata)
    
    def add_citation(self, citing_paper: str, cited_paper: str):
        """Add citation edge to network"""
        if citing_paper in self.graph.nodes() and cited_paper in self.graph.nodes():
            self.graph.add_edge(citing_paper, cited_paper)
    
    def calculate_influence_metrics(self) -> Dict:
        """Calculate various influence metrics"""
        if len(self.graph.nodes()) == 0:
            return {}
        
        metrics = {}
        
        try:
            # PageRank for global influence
            pagerank = nx.pagerank(self.graph) if len(self.graph.edges()) > 0 else {}
            
            # Citation count (in-degree)
            citation_counts = dict(self.graph.in_degree())
            
            # Betweenness centrality for bridging different areas
            betweenness = nx.betweenness_centrality(self.graph) if len(self.graph.edges()) > 0 else {}
            
            # Combine metrics
            max_citations = max(citation_counts.values()) if citation_counts.values() else 1
            
            for node in self.graph.nodes():
                metrics[node] = {
                    'pagerank': pagerank.get(node, 0),
                    'citations': citation_counts.get(node, 0),
                    'betweenness': betweenness.get(node, 0),
                    'influence_score': (
                        0.5 * pagerank.get(node, 0) + 
                        0.3 * (citation_counts.get(node, 0) / max_citations) +
                        0.2 * betweenness.get(node, 0)
                    )
                }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            # Fallback metrics
            for node in self.graph.nodes():
                metrics[node] = {
                    'pagerank': 0,
                    'citations': 0,
                    'betweenness': 0,
                    'influence_score': 0.1
                }
        
        return metrics
    
    def identify_research_trends(self, time_window: int = 5) -> Dict:
        """Identify emerging research trends"""
        current_year = datetime.now().year
        trends = {}
        
        # Analyze papers from recent years
        recent_papers = [
            node for node in self.graph.nodes()
            if self.graph.nodes[node].get('year', 0) >= current_year - time_window
        ]
        
        # Extract keywords from titles and abstracts
        keywords = {}
        for paper in recent_papers:
            title = self.graph.nodes[paper].get('title', '').lower()
            abstract = self.graph.nodes[paper].get('abstract', '').lower()
            text = title + ' ' + abstract
            
            # Extract meaningful words
            words = re.findall(r'\b[a-z]{4,}\b', text)
            common_words = {'paper', 'study', 'research', 'analysis', 'method', 'approach', 'results', 'conclusion', 'introduction'}
            
            for word in words:
                if word not in common_words and len(word) > 3:
                    keywords[word] = keywords.get(word, 0) + 1
        
        # Sort by frequency to identify trends
        trends = dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:20])
        
        return trends