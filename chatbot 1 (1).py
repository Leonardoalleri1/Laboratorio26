import streamlit as st
import pdfplumber

# --- LIBRERIE ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. CONFIGURAZIONE PAGINA
st.set_page_config(page_title="Assistenza ERSU", page_icon=":classical_building:")

# 2. STILE E GRAFICA (CSS MODERNO)
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #2A64C5, #3B7BFF); color: #EAF6FF; }
    
    /* Titolo con gradiente */
    .gradient-text {
        background: linear-gradient(to right, #ffffff, #a0c4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 0px;
    }
    
    .motto { text-align: center; font-style: italic; color: #EAF6FF; margin-bottom: 25px; }
    
    /* Effetto floating */
    .floating-logo {
        animation: float 4s ease-in-out infinite;
        filter: drop-shadow(0 10px 15px rgba(0,0,0,0.2));
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-15px); }
        100% { transform: translateY(0px); }
    }
    
    /* Glassmorphism */
    .main .block-container { 
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 30px;
        padding: 2rem;
        margin-top: 2rem;
    }
    
    .footer { 
        position: fixed; left: 0; bottom: 0; width: 100%; text-align: center; 
        background: rgba(255, 255, 255, 0.15); backdrop-filter: blur(10px); 
        padding: 15px; z-index: 999; color: white;
    }
</style>
""", unsafe_allow_html=True)

# 3. INTERFACCIA UTENTE
st.markdown("<h1 class='gradient-text'>Assistenza ERSU</h1>", unsafe_allow_html=True)
st.markdown("<p class='motto'>Il tuo supporto intelligente per la vita universitaria</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="floating-logo">', unsafe_allow_html=True)
    st.image("Chatbot (1).webp", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 4. LOGICA CHATBOT
documento = "Costituzione_italiana.pdf"

# [Qui inserisci la tua logica di caricamento PDF, embedding e chain che avevi prima]
# Se il PDF non viene trovato, assicurati che il file sia nella stessa cartella del .py
if documento:
    # Aggiungi qui il resto del tuo codice originale...
    st.info("Logica chatbot attiva")

# 5. FOOTER
st.markdown("""
<div class="footer">
    <p><strong>ERSU Palermo</strong> | <a href="https://www.ersupalermo.it/" style="color:white;">Sito Ufficiale</a></p>
</div>
""", unsafe_allow_html=True)

