Academic-RAG-System

Academic Paper RAG with Citation Network system that not only retrieves relevant academic papers but also understands citation relationships, research trends, and can identify seminal works in a field.

1. Prerequisites

* Python 3.9+, VS Code, Git installed
* Required VS Code extensions (Python, Pylance, etc.)

2. Project Setup

* Create project folder structure
* Set up virtual environment
* Install dependencies from requirements.txt

3. Code Implementation

* Modular architecture with separate files for each component
* Main app.py with Streamlit interface
* Source modules for paper processing, citation analysis, and RAG

4. Running the Application

bash

       # Activate virtual environment
         venv\Scripts\activate

       # Install dependencies  
         pip install -r requirements.txt

       # Run the app
        streamlit run app.py


<img width="1920" height="1080" alt="Screenshot 2025-08-16 003052" src="https://github.com/user-attachments/assets/2d598e0e-0ac0-4030-b7d5-021a3df306e2" />

6. Key Features Ready to Use

📤 PDF upload and processing
🔍 Semantic search with citation context
🕸️ Citation network visualization
📈 Research trends analysis
⭐ Influence scoring

🔧 Important Notes for VS Code

Virtual Environment: Make sure VS Code uses the correct Python interpreter from your venv folder
Extensions: Install the recommended Python extensions for better development experience
Terminal: Use VS Code's integrated terminal for all commands
Debugging: Use F5 to debug or create launch configurations

🚨 Common Issues & Solutions

Import errors: Ensure virtual environment is activated
ChromaDB issues: Try installing with --no-cache-dir flag
PDF processing errors: Install alternative libraries like pdfplumber
Port conflicts: Use different port with --server.port 8502

📊 Testing Your Deployment

Upload sample PDF papers
Try search queries like "machine learning" or "data analysis"
Check citation network visualization
Explore research trends page



