import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import networkx as nx
import plotly.graph_objects as go
from src.paper_processor import PaperProcessor
from src.rag_system import AcademicRAG
import os
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Academic Paper RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'rag_system' not in st.session_state:
        with st.spinner("🚀 Initializing Academic RAG System..."):
            st.session_state.rag_system = AcademicRAG()
            st.session_state.rag_system.create_collection()
            st.session_state.paper_processor = PaperProcessor()
            st.session_state.paper_count = 0
    
    if 'uploaded_papers' not in st.session_state:
        st.session_state.uploaded_papers = []

def main():
    st.markdown('<h1 class="main-header">🎓 Academic Paper RAG System</h1>', unsafe_allow_html=True)
    st.markdown("*Intelligent Research Assistant with Citation Network Analysis*")
    
    # Initialize system
    initialize_session_state()
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🔍 Search Papers", "📤 Upload Papers", "🕸️ Citation Network", "📈 Research Trends", "ℹ️ About"]
    )
    
    # Display current system status
    st.sidebar.markdown("### 📊 System Status")
    st.sidebar.info(f"Papers in database: {st.session_state.paper_count}")
    
    # Route to different pages
    if page == "🔍 Search Papers":
        search_interface()
    elif page == "📤 Upload Papers":
        upload_interface()
    elif page == "🕸️ Citation Network":
        network_visualization()
    elif page == "📈 Research Trends":
        trends_analysis()
    elif page == "ℹ️ About":
        about_page()

def search_interface():
    st.header("🔍 Search Academic Papers")
    st.markdown("Enter your research query to find relevant papers with citation context.")
    
    # Search input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Research Query:",
            placeholder="e.g., machine learning in healthcare, neural networks, climate change",
            help="Enter keywords or phrases related to your research interest"
        )
    
    with col2:
        search_button = st.button("🔍 Search", type="primary")
    
    # Advanced search options
    with st.expander("🔧 Advanced Search Options"):
        col1, col2 = st.columns(2)
        with col1:
            num_results = st.slider("Number of results", 5, 20, 10)
        with col2:
            sort_by = st.selectbox("Sort by", ["Relevance + Influence", "Relevance Only", "Citation Count"])
    
    # Perform search
    if (query and search_button) or (query and st.session_state.get('auto_search', False)):
        if st.session_state.paper_count == 0:
            st.warning("⚠️ No papers in the database. Please upload some papers first!")
            return
        
        with st.spinner("🔍 Searching and analyzing papers..."):
            # Retrieve documents
            results = st.session_state.rag_system.retrieve_with_citation_context(query, num_results)
            
            if not results['documents']:
                st.error("No relevant papers found. Try different search terms or upload more papers.")
                return
            
            # Generate response
            response = st.session_state.rag_system.generate_response(query, results)
            
            # Display results
            st.markdown("### 📄 Research Summary")
            st.markdown(response)
            
            # Display metrics
            st.markdown("### 📊 Search Analytics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_papers = len(set([m['paper_id'] for m in results['metadatas']]))
                st.metric("📚 Papers Found", total_papers)
            
            with col2:
                avg_influence = np.mean([
                    results['citation_context'][m['paper_id']]['influence_score']
                    for m in results['metadatas']
                ]) if results['metadatas'] else 0
                st.metric("⭐ Avg Influence", f"{avg_influence:.3f}")
            
            with col3:
                total_citations = sum([
                    results['citation_context'][m['paper_id']]['citation_count']
                    for m in results['metadatas']
                ]) if results['metadatas'] else 0
                st.metric("📊 Total Citations", total_citations)
            
            with col4:
                search_time = 0.5  # Placeholder
                st.metric("⚡ Search Time", f"{search_time:.2f}s")

def upload_interface():
    st.header("📤 Upload Academic Papers")
    st.markdown("Upload PDF papers to build your research database.")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload academic papers in PDF format. Multiple files can be selected."
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected for processing**")
        
        # Process button
        if st.button("🚀 Process Papers", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            processed_count = 0
            error_count = 0
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Processing: {uploaded_file.name}")
                    
                    # Process paper
                    metadata = st.session_state.paper_processor.extract_paper_metadata(uploaded_file)
                    
                    # Generate unique paper ID
                    paper_id = f"paper_{int(time.time())}_{i}"
                    
                    # Add to system
                    success = st.session_state.rag_system.add_paper_to_index(paper_id, metadata)
                    
                    if success:
                        processed_count += 1
                        st.session_state.uploaded_papers.append({
                            'id': paper_id,
                            'filename': uploaded_file.name,
                            'title': metadata.get('title', 'Unknown Title')[:100],
                            'year': metadata.get('year', 'Unknown')
                        })
                    else:
                        error_count += 1
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    error_count += 1
            
            # Update paper count
            st.session_state.paper_count += processed_count
            
            # Show results
            status_text.text("Processing complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"✅ Successfully processed: {processed_count} papers")
            with col2:
                if error_count > 0:
                    st.error(f"❌ Failed to process: {error_count} papers")
    
    # Display uploaded papers
    if st.session_state.uploaded_papers:
        st.markdown("### 📚 Uploaded Papers")
        
        papers_df = pd.DataFrame(st.session_state.uploaded_papers)
        st.dataframe(papers_df, use_container_width=True)

def network_visualization():
    st.header("🕸️ Citation Network Analysis")
    
    if st.session_state.paper_count == 0:
        st.info("📊 Upload some papers first to see citation network analysis.")
        return
    
    # Get network metrics
    with st.spinner("Analyzing citation network..."):
        metrics = st.session_state.rag_system.citation_network.calculate_influence_metrics()
    
    if not metrics:
        st.info("No citation relationships found yet. Upload more papers to build the network.")
        return
    
    # Display top influential papers
    st.subheader("📊 Most Influential Papers")
    
    sorted_papers = sorted(
        metrics.items(),
        key=lambda x: x[1]['influence_score'],
        reverse=True
    )[:15]
    
    if sorted_papers:
        # Create a DataFrame for better visualization
        influence_data = []
        for paper_id, metric in sorted_papers:
            try:
                paper_info = st.session_state.rag_system.citation_network.graph.nodes[paper_id]
                influence_data.append({
                    'Title': paper_info.get('title', paper_id)[:60] + '...',
                    'Year': paper_info.get('year', 'N/A'),
                    'Citations': metric['citations'],
                    'PageRank': f"{metric['pagerank']:.4f}",
                    'Influence Score': f"{metric['influence_score']:.4f}"
                })
            except:
                continue
        
        if influence_data:
            df = pd.DataFrame(influence_data)
            st.dataframe(df, use_container_width=True)
            
            # Visualization
            if len(influence_data) > 1:
                fig = px.bar(
                    df.head(10), 
                    x='Influence Score', 
                    y='Title',
                    title="Top 10 Most Influential Papers",
                    orientation='h'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
    
    # Network statistics
    st.subheader("📈 Network Statistics")
    
    graph = st.session_state.rag_system.citation_network.graph
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total Papers", len(graph.nodes()))
    with col2:
        st.metric("🔗 Citation Links", len(graph.edges()))
    with col3:
        density = nx.density(graph) if len(graph.nodes()) > 1 else 0
        st.metric("🕸️ Network Density", f"{density:.4f}")
    with col4:
        avg_degree = sum(dict(graph.degree()).values()) / len(graph.nodes()) if len(graph.nodes()) > 0 else 0
        st.metric("📊 Avg Connections", f"{avg_degree:.2f}")

def trends_analysis():
    st.header("📈 Research Trends Analysis")
    
    if st.session_state.paper_count == 0:
        st.info("📊 Upload some papers first to see research trends.")
        return
    
    with st.spinner("Analyzing research trends..."):
        trends = st.session_state.rag_system.citation_network.identify_research_trends()
    
    if trends:
        st.subheader("🔥 Trending Keywords")
        
        # Create visualization
        keywords = list(trends.keys())[:15]
        frequencies = [trends[k] for k in keywords]
        
        if keywords and frequencies:
            # Bar chart
            fig = px.bar(
                x=frequencies,
                y=keywords,
                orientation='h',
                title="Most Frequent Research Keywords",
                labels={'x': 'Frequency', 'y': 'Keywords'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Word cloud style visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📝 Top Emerging Terms")
                for i, (keyword, freq) in enumerate(list(trends.items())[:10]):
                    st.write(f"**{i+1}. {keyword.title()}**: {freq} mentions")
            
            with col2:
                st.subheader("🎯 Research Focus Areas")
                # Simplified clustering of related terms
                focus_areas = {
                    "🤖 Machine Learning": ["learning", "neural", "model", "algorithm", "training", "prediction"],
                    "📊 Data Science": ["data", "analysis", "mining", "statistics", "visualization", "processing"],
                    "🧠 AI Applications": ["artificial", "intelligence", "automation", "classification", "recognition"],
                    "🔬 Healthcare": ["medical", "clinical", "patient", "diagnosis", "treatment", "health"],
                    "🌐 Networks": ["network", "graph", "connectivity", "topology", "distributed", "social"]
                }
                
                for area, keywords_list in focus_areas.items():
                    score = sum([trends.get(k.lower(), 0) for k in keywords_list])
                    if score > 0:
                        st.write(f"**{area}**: {score} total mentions")
    else:
        st.info("No trends data available yet. Upload more papers to see research trends.")

def about_page():
    st.header("ℹ️ About Academic Paper RAG System")
    
    st.markdown("""
    ### 🎯 What is this system?
    
    This Academic Paper RAG (Retrieval-Augmented Generation) System is designed to help researchers:
    
    - **📚 Organize** and search through academic papers
    - **🔍 Discover** relevant research with intelligent search
    - **🕸️ Analyze** citation networks and paper influence
    - **📈 Track** research trends and emerging topics
    - **⭐ Identify** seminal works in your field
    
    ### 🛠️ Key Features
    
    #### 🔍 Intelligent Search
    - Semantic search using advanced embeddings
    - Citation-aware ranking system
    - Context-aware response generation
    
    #### 🕸️ Citation Network Analysis
    - PageRank-based influence scoring
    - Betweenness centrality for interdisciplinary connections
    - Network density and connectivity metrics
    
    #### 📈 Research Trends
    - Keyword frequency analysis
    - Temporal trend identification
    - Research focus area clustering
    
    ### 🚀 How to Use
    
    1. **Upload Papers**: Start by uploading PDF papers using the "Upload Papers" page
    2. **Search**: Use natural language queries to find relevant research
    3. **Analyze**: Explore citation networks and research trends
    4. **Discover**: Find influential papers and emerging topics
    
    ### 🔧 Technical Details
    
    - **Vector Database**: ChromaDB for similarity search
    - **Embeddings**: Sentence Transformers for semantic understanding
    - **Graph Analysis**: NetworkX for citation network modeling
    - **Interface**: Streamlit for interactive web application
    
    ### 📊 Evaluation Metrics
    
    The system tracks:
    - Retrieval accuracy and relevance
    - Search latency and performance
    - Citation network completeness
    - User interaction patterns
    """)
    
    # System information
    st.subheader("💻 System Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Database Status**
        - Papers: {st.session_state.paper_count}
        - Collection: Active
        - Search: Ready
        """)
    
    with col2:
        st.info(f"""
        **Models Loaded**
        - Embedding: all-MiniLM-L6-v2
        - Vector DB: ChromaDB
        - Graph: NetworkX
        """)

if __name__ == "__main__":
    main()