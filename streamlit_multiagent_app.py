import streamlit as st
import os
import io
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# AI and LangChain Imports
import google.generativeai as genai
from langdetect import detect, LangDetectException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ============================================================================
# 1. PERSONA DEFINITIONS & CONFIGURATIONS
# ============================================================================

class PersonaType(Enum):
    FITNESS = "fitness"
    MENTAL_HEALTH = "mental_health"
    FINANCE = "finance"
    IMMIGRATION = "immigration"
    PARENTING = "parenting"

@dataclass
class PersonaConfig:
    name: str
    system_prompt: str
    keywords: List[str]
    rag_corpus: List[str]
    safety_rules: List[str]

SUPPORTED_LANGUAGES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese',
    'ja': 'Japanese', 'hi': 'Hindi', 'ar': 'Arabic', 'ko': 'Korean'
}

# FULL DATA FROM NOTEBOOK
PERSONAS = {
    PersonaType.FITNESS.name: PersonaConfig(
        name="Fitness Coach",
        system_prompt="""You are an experienced and supportive fitness coach. You help users with:
        - Workout planning and exercise recommendations
        - Basic nutrition education and healthy eating habits
        - Habit building and motivation for fitness goals
        - Form and technique guidance.
        You provide evidence-based advice and encourage sustainable, healthy practices. Always consider the user's fitness level.""",
        keywords=['workout', 'exercise', 'fitness', 'gym', 'training', 'muscle', 'cardio', 'nutrition', 'diet', 'protein', 'calories', 'weight loss', 'strength'],
        rag_corpus=[
            "Cardiovascular exercise improves heart health. Aim for 150 minutes of moderate-intensity or 75 minutes of vigorous-intensity aerobic activity per week.",
            "Strength training should be performed 2-3 times per week, targeting all major muscle groups. Allow 48 hours between sessions for muscle recovery.",
            "Proper form is crucial for preventing injuries. Start with lighter weights and focus on technique before increasing resistance.",
            "Protein intake of 0.8-1.0g per kg of body weight supports general health. Athletes may need 1.2-2.0g per kg for muscle building.",
            "Hydration is essential for performance. Drink water before, during, and after exercise. Aim for 8-10 glasses daily.",
            "Progressive overload - gradually increasing weight, reps, or intensity - is key to continued improvement.",
            "Rest and recovery are as important as training. Get 7-9 hours of sleep and take rest days to allow muscles to repair.",
            "Compound exercises like squats, deadlifts, and bench press work multiple muscle groups and are efficient for building strength."
        ],
        safety_rules=["Never diagnose medical conditions", "Recommend seeing a doctor before starting new intense programs", "Avoid extreme diet advice"]
    ),
    PersonaType.MENTAL_HEALTH.name: PersonaConfig(
        name="Mental Health Counsellor",
        system_prompt="""You are a supportive and empathetic mental health counsellor. 
        CRITICAL: You are NOT a licensed therapist. You provide emotional support, active listening, and coping strategies.
        Tone: warm, non-judgmental, and encouraging.""",
        keywords=['anxiety', 'depression', 'stress', 'therapy', 'mental health', 'panic', 'worry', 'sad', 'overwhelmed', 'burnout', 'crisis'],
        rag_corpus=[
            "Deep breathing (4-7-8 technique) can help activate the parasympathetic nervous system and reduce acute anxiety symptoms.",
            "Cognitive reframing involves identifying negative thought patterns and replacing them with more balanced, evidence-based alternatives.",
            "Regular physical activity has been shown to reduce symptoms of anxiety and depression by improving mood-regulating neurotransmitters.",
            "Sleep hygiene basics: consistent bedtime, limit screens 1 hour before sleep, cool/dark room, avoid caffeine late in the day.",
            "If someone is in immediate danger or considering self-harm, encourage contacting emergency services or local crisis hotlines immediately."
        ],
        safety_rules=["Always include disclaimer: not a licensed therapist", "Provide crisis resources if self-harm/suicidal ideation appears", "Avoid diagnosing conditions"]
    ),
    PersonaType.FINANCE.name: PersonaConfig(
        name="Personal Finance Educator",
        system_prompt="""You are a practical personal finance educator. You help with budgeting, debt payoff, and saving.
        IMPORTANT: You do NOT provide personalized investment advice. You provide general education only.""",
        keywords=['budget', 'saving', 'debt', 'credit', 'invest', '401k', 'ira', 'loan', 'interest', 'retirement', 'stocks'],
        rag_corpus=[
            "A common budgeting framework is 50/30/20: 50% needs, 30% wants, 20% savings/debt repayment.",
            "An emergency fund typically covers 3-6 months of essential expenses; start with a smaller goal like $500-$1,000.",
            "Debt payoff methods: avalanche (highest interest first) minimizes total interest; snowball (smallest balance first) builds momentum.",
            "Index funds provide broad diversification and typically have low expense ratios compared to actively managed funds.",
            "Credit score factors include payment history, utilization, length of credit history, new credit, and credit mix."
        ],
        safety_rules=["Do not provide personalized investment advice", "Explain risks and uncertainty", "Encourage professional advice for major decisions"]
    ),
    PersonaType.IMMIGRATION.name: PersonaConfig(
        name="Immigration Guide",
        system_prompt="""You are an immigration information guide. You help users understand pathways and terminology.
        IMPORTANT: You are NOT a lawyer and do NOT provide legal advice. Always recommend consulting an attorney.""",
        keywords=['visa', 'immigration', 'green card', 'uscis', 'asylum', 'citizenship', 'work permit', 'h1b', 'f1'],
        rag_corpus=[
            "Family-based immigration allows US citizens and permanent residents to petition for certain family members.",
            "Employment-based immigration has five preference categories (EB-1 through EB-5).",
            "Asylum may be granted to those who fear persecution based on race, religion, nationality, or political opinion.",
            "USCIS (United States Citizenship and Immigration Services) is the primary agency handling immigration benefits.",
            "Processing times vary significantly by visa type. Check official USCIS processing times for current estimates."
        ],
        safety_rules=["ALWAYS clarify you provide general information, not legal advice", "Direct users to official government resources", "Never make case-specific recommendations"]
    ),
    PersonaType.PARENTING.name: PersonaConfig(
        name="Parenting Guide",
        system_prompt="""You are a supportive parenting guide helping with childcare and development.
        IMPORTANT: You do NOT provide medical diagnoses. Always recommend consulting pediatricians for health concerns.""",
        keywords=['baby', 'child', 'parenting', 'kids', 'toddler', 'discipline', 'development', 'feeding', 'behavior', 'school'],
        rag_corpus=[
            "Newborns typically sleep 14-17 hours per day in short periods.",
            "Positive reinforcement is more effective than punishment for shaping behavior.",
            "Reading to children daily supports language development and literacy skills.",
            "Screen time recommendations: Under 18 months, avoid except video chatting. Ages 2-5, limit to 1 hour daily.",
            "Developmental milestones vary: most walk by 15 months and speak in 2-word phrases by 24 months.",
            "Consistent routines help children feel secure. Bedtime routines are especially important for sleep.",
            "Emotional validation helps children develop emotional intelligence.",
            "Play is crucial for development, building creativity and social skills."
        ],
        safety_rules=["Never provide medical diagnoses", "Recommend consulting pediatricians for health issues", "Be non-judgmental"]
    )
}

# ============================================================================
# 2. CORE SYSTEM LOGIC (RAG & ROUTING)
# ============================================================================

class RAGManager:
    def __init__(self):
        # Open-source embeddings as per notebook
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_stores = {}
        self._initialize_stores()

    def _initialize_stores(self):
        for p_name, cfg in PERSONAS.items():
            docs = [Document(page_content=d) for d in cfg.rag_corpus]
            self.vector_stores[p_name] = Chroma.from_documents(
                documents=docs, 
                embedding=self.embeddings, 
                collection_name=f"col_{p_name.lower()}"
            )

    def retrieve(self, persona_name: str, query: str):
        if persona_name not in self.vector_stores: return []
        docs = self.vector_stores[persona_name].similarity_search(query, k=3)
        return [d.page_content for d in docs]

class RouterAgent:
    def route(self, query: str) -> str:
        query = query.lower()
        scores = {}
        for p_name, cfg in PERSONAS.items():
            scores[p_name] = sum(1 for k in cfg.keywords if k in query)
        
        best_p = max(scores, key=scores.get)
        # Default to Fitness if no keywords match
        return best_p if scores[best_p] > 0 else PersonaType.FITNESS.name

# ============================================================================
# 3. STREAMLIT UI & SESSION STATE
# ============================================================================

st.set_page_config(page_title="Multi-Agent AI Assistant", layout="centered", page_icon="🤖")

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatMessage { border-radius: 15px; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ System Settings")
    api_key = st.text_input("Enter Google API Key", type="password", help="Needed to run Gemini 2.0")
    st.divider()
    
    st.subheader("🤖 Agent Configuration")
    routing_mode = st.radio("Selection Mode", ["Smart Auto-Route", "Manual Override"])
    manual_agent = st.selectbox("Select Agent (Manual)", list(PERSONAS.keys()))
    
    st.divider()
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Persistent Backend (Prevents reloading embeddings on every run)
if "rag_engine" not in st.session_state:
    with st.spinner("Initializing Knowledge Base..."):
        st.session_state.rag_engine = RAGManager()
        st.session_state.router_engine = RouterAgent()

st.title("🌟 Multi-Agent Assistant")
st.caption("Specialized expertise in Fitness, Mental Health, Finance, Immigration, and Parenting.")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("How can I help you today?"):
    if not api_key:
        st.error("Please provide a Google API Key in the sidebar.")
        st.stop()
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    # User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Response
    with st.chat_message("assistant"):
        # 1. Detect Language
        try:
            lang_code = detect(prompt)
            lang_name = SUPPORTED_LANGUAGES.get(lang_code, "English")
        except:
            lang_name = "English"

        # 2. Route to Persona (Using string-based keys to prevent KeyError)
        if routing_mode == "Smart Auto-Route":
            active_persona_key = st.session_state.router_engine.route(prompt)
        else:
            active_persona_key = manual_agent
        
        persona = PERSONAS[active_persona_key]
        st.caption(f"**Agent:** {persona.name} | **Language:** {lang_name}")

        # 3. Retrieve Context (RAG)
        context_chunks = st.session_state.rag_engine.retrieve(active_persona_key, prompt)
        context_text = "\n".join(context_chunks)

        # 4. Generate Response with Gemini 2.0 Flash Lite
        try:
            # Using the high-rate-limit model recommended in your notebook
            model = genai.GenerativeModel("gemini-2.0-flash-lite")
            
            system_instruction = f"""
            Role: {persona.system_prompt}
            Context Information: {context_text}
            Safety Constraints: {persona.safety_rules}
            Instructions: Answer the query using the context provided. Respond in {lang_name}.
            """
            
            response = model.generate_content([system_instruction, prompt])
            ans_text = response.text

            # 5. Apply Post-Processing Safety Logic (From Section 11 of notebook)
            if active_persona_key == PersonaType.MENTAL_HEALTH.name:
                if any(word in prompt.lower() for word in ['suicide', 'hurt', 'harm', 'kill']):
                    ans_text += "\n\n🚨 **CRISIS RESOURCES**: Please reach out to the 988 Suicide & Crisis Lifeline (USA) or your local emergency services immediately."
                else:
                    ans_text += "\n\n*Disclaimer: I am an AI assistant, not a licensed mental health professional.*"
            
            elif active_persona_key == PersonaType.IMMIGRATION.name:
                ans_text += "\n\n*Note: This information is for educational purposes and is not legal advice.*"

            st.markdown(ans_text)
            st.session_state.messages.append({"role": "assistant", "content": ans_text})

        except Exception as e:
            st.error(f"Generation Error: {str(e)}")