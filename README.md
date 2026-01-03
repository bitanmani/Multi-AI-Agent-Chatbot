# Multi-AI-Agent-Chatbot
## 🎬 Demo
https://youtu.be/xp_Iop2NJY0

This is a sophisticated, production-ready AI framework built on Google Gemini 2.0 Flash Lite. It utilizes a Multi-Agent System (MAS) architecture combined with Retrieval-Augmented Generation (RAG) to provide expert-level assistance across five specialized domains: Fitness, Mental Health, Personal Finance, Immigration, and Parenting.

The system is designed with a "Safety-First" philosophy, incorporating domain-specific guardrails and automated language detection to serve a global user base through both Text and Voice interfaces.

🏗️ System Architecture & Workflow
The project follows a modular, decoupled architecture that separates intent recognition, knowledge retrieval, and response synthesis.

Input Layer: Captures user input via Text or Speech-to-Text (STT).

Language Orchestrator: Automatically detects the input language (supporting 12+ ISO codes) using langdetect to ensure the entire response pipeline remains consistent with the user's language.

Router Agent: Performs semantic analysis to map the user's query to one of the five specialized Persona Agents.

Agentic RAG Pipeline:

The selected agent queries its own isolated ChromaDB collection.

Embeddings: Uses sentence-transformers/all-MiniLM-L6-v2 for efficient local vectorization.

Contextual Retrieval: Fetches top-K relevant documents from a curated knowledge corpus.

Reasoning & Tools: Agents can execute domain-specific tools (e.g., therapist_finder, official_docs_search) to provide actionable resources beyond just text.

Synthesis Layer: Gemini 2.0 Flash Lite synthesizes the system prompt, retrieved context, and tool outputs into a final response.

Output Layer: Delivers the response via Text or Text-to-Speech (TTS).

🧠 Core Component Breakdown
1. The Multi-Agent Layer
Rather than a general-purpose chatbot, Sentinel-AI utilizes specialized agents defined via PersonaConfig:

Fitness Coach: Focuses on workout planning and nutrition.

Mental Health Counsellor: Provides empathetic support with strict crisis-detection rules.

Personal Finance Educator: Offers budgeting and debt-reduction strategies.

Immigration Guide: Navigates complex visa and policy pathways.

Parenting Guide: Covers developmental milestones and behavioral strategies.

2. The RAG Engine (RAGManager)
The system implements a robust RAG pipeline using LangChain and ChromaDB:

Vector Isolation: Each persona has a dedicated vector store, preventing "cross-domain contamination" where finance data might bleed into health advice.

Recursive Chunking: Implements RecursiveCharacterTextSplitter to ensure context remains semantically meaningful during retrieval.

3. Tool Executor Framework
Agents are "agentic" because they can interact with the world:

web_search: General information gathering.

official_docs_search: Specialized for USCIS and legal resources.

document_analyzer: Simulates financial data parsing for the Finance agent.

🛠️ Technical Stack & Skills
AI & Machine Learning
LLM Orchestration: LangChain (Core, Community, Google GenAI).

Models: Google Gemini 2.0 Flash Lite (Generative), HuggingFace all-MiniLM-L6-v2 (Embeddings).

Vector Search: ChromaDB for persistence and similarity search.

NLP Tools: langdetect for multilingual routing.

Software Engineering
Design Patterns: Modular Agent-based design, Factory patterns for persona initialization.

Data Architecture: Persistent vector databases and structured dataclass configurations.

Multimodal: Integration of gTTS and SpeechRecognition for accessibility.

🚦 Installation & Setup
1. Clone and Install Dependencies
pip install google-generativeai langchain langchain-chroma chromadb \
            sentence-transformers langdetect gtts SpeechRecognition \
            python-dotenv pydub

2. Configure API Keys
Create a .env file or set your environment variable:
export GOOGLE_API_KEY="your_actual_gemini_api_key"

3. Initialize the System
The system automatically builds the vector stores on the first run using the pre-defined expert corpora.

🛡️ Safety & Ethics
This project prioritizes Responsible AI:

Disclaimers: Every health, legal, or financial response includes mandatory disclaimers.

Crisis Protocol: The Mental Health agent is hard-coded to prioritize 988/Crisis resources if self-harm signals are detected.

Evidence-Based: Agents are instructed to prioritize retrieved RAG context over model-only "hallucinations".

📈 Future Scope
Implement Redis for persistent user session memory.

Integrate Tavily API for real-time live web search.

Deploy a Streamlit or FastAPI dashboard for production monitoring.
