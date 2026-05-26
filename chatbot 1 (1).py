# ###########################################################
# CHATBOT SENZA IL CARICAMENTO DEL PDF DA PARTE DELL'UTENTE #
# ###########################################################

import streamlit as st
import pdfplumber

# Langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Elenco di tutte le icone Streamlit:
# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
st.set_page_config(page_title= "RagChatbot",
                   page_icon=":classical_building:")

# Personalizzazione colori:
# Colori esadecimali: https://divmagic.com/it/tools/color-converter
st.markdown("""

<style>

/* =========================

   SFONDO PRINCIPALE

========================= */

.stApp {

    background: linear-gradient(

        135deg,

        #2A64C5,

        #3B7BFF

    );

    color: #EAF6FF;

}

/* =========================

   TITOLO

========================= */

h1, h2, h3 {

    color: #EAF6FF;

    font-weight: 700;

}

/* =========================

   LASTRA CENTRALE

========================= */

.main .block-container {

    background: rgba(255, 255, 255, 0.10);

    border: 1px solid rgba(255, 255, 255, 0.25);

    backdrop-filter: blur(18px);

    border-radius: 28px;

    padding: 2rem;

    margin-top: 2rem;

    box-shadow: 0 10px 40px rgba(0,0,0,0.15);

}

/* =========================

   INPUT CHAT

========================= */

.stTextInput input {

    background-color: #1a2333; 
    color: #ffffff;           
    border: 1px solid #3b7bff; 
    border-radius: 14px;
    padding: 12px;
    font-size: 16px;

}

/* FOCUS INPUT */

.stTextInput input:focus {

    background-color: #1e2a3d; 
    border-color: #ffffff;     
    box-shadow: 0 0 8px rgba(59, 123, 255, 0.5);
    outline: none

}
div[data-testid="stHorizontalBlock"] {
    align-items: flex-end;
    gap: 10px;
}

.stButton button {
    background-color: #1a2333 !important;
    color: white !important;
    border: 1px solid #3b7bff !important;
    border-radius: 14px;
    padding: 10px 20px;
    font-weight: bold;
    height: 44px; /* Fondamentale per allinearlo all'input */
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.03);
    background-color: #2A64C5 !important;
}

/* =========================

   BOTTONI

========================= */

.stButton button {

    background: white;

    color: #2A64C5;

    border: none;

    border-radius: 14px;

    padding: 10px 20px;

    font-weight: bold;

    transition: 0.3s;

}

.stButton button:hover {

    transform: scale(1.03);

}

/* =========================

   RISPOSTA

========================= */

.stMarkdown {

    color: white;

}

/* =========================

   PARALLASSE SIMULATO

========================= */

.stApp::before {

    content: "";

    position: fixed;

    top: 0;

    left: 0;

    right: 0;

    bottom: 0;

    background: radial-gradient(

        circle at top,

        rgba(255,255,255,0.18),

        transparent 60%

    );

    animation: parallaxMove 12s ease-in-out infinite alternate;

    pointer-events: none;

    z-index: 0;

}

@keyframes parallaxMove {

    0% { transform: translateY(0px); }

    100% { transform: translateY(-25px); }

}

/* =========================

   SIDEBAR

========================= */

section[data-testid="stSidebar"] {

    background-color: rgba(255,255,255,0.08);

    border-right: 1px solid rgba(255,255,255,0.2);

}

</style>

""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #EAF6FF;">Assistenza ERSU Palermo</h1>
        <p style="font-size: 1.2rem; font-style: italic; color: #EAF6FF;">
            Il tuo supporto intelligente per la vita universitaria
        </p>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image("ERSU.AI2-Photoroom.png", use_container_width=True)
documento = "Costituzione_italiana.pdf"

# Estrazione del contenuto e spezzettamento
if documento is not None:
    @st.cache_data(show_spinner="Sto leggendo il PDF...")
    def estrai_testo_pdf(documento: str) -> str:
        with pdfplumber.open(documento) as pdf:
            # st.write(f"Pagine totali: {len(pdf.pages)} - Comincio la scansione...")
            testo = ""
            for pagina in pdf.pages:
                # Se la pagina è null menttiamo ""
                testo_pagina = pagina.extract_text() or ""
                testo = testo + testo_pagina + "\n"
                # testo += pagina.extract_text() + "\n"
        return testo.strip()
    
    testo = estrai_testo_pdf(documento)

    @st.cache_data(show_spinner=False)
    def crea_frammenti(testo: str):
        taglierina = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=1000,
        chunk_overlap=200)
        return taglierina.split_text(testo)

    frammenti = crea_frammenti(testo)
    # st.write(f"Totale frammenti creati: {len(frammenti)}")
    # st.write(frammenti)

    # Generiamo gli embeddings
    # e li salviamo in un vector store o vector db (es. FAISS, Pinecone, etc.)
    # Puoi cambiare OpenAIEmbeddings e metterne altri
    # https://docs.langchain.com/oss/python/integrations/embeddings
    @st.cache_resource(show_spinner=False)
    def crea_vectorstore(frammenti):
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=st.secrets["OPENAI_API_KEY"])
        return FAISS.from_texts(frammenti, embedding=embeddings)
    
    vettori = crea_vectorstore(frammenti)
    # st.write("Embedding recuperati!")

    # -------------------------------------------------------------------
    # Gestione prompt
    # -------------------------------------------------------------------
    # def invia():
        # st.session_state.domanda_inviata = st.session_state.domanda_utente
        # salva il contenuto di input, cioè domanda_utente, in domanda_inviata
        # st.session_state.domanda_utente = ""
        # reset dopo invio

    # st.text_input("Chiedi al chatbot:", key="domanda_utente", on_change=invia)
    # key="domanda_utente": assegna a st.session_state ciò che scriviamo (domanda_utente)
    # Ogni volta che l’utente modifica il campo e preme Invio,
    # la funzione invia() viene chiamata.

    # domanda_utente = st.session_state.get("domanda_inviata", "")
    # Recupera il valore salvato in "domanda_inviata".
    # Se "domanda_inviata" non è ancora stato definito (es. al primo avvio dell'app),
    # allora il valore predefinito sarà "" (secondo argomento dell'istruzione)
    # --------------------------------------------------

    def invia():
        st.session_state.domanda_inviata = st.session_state.domanda_utente
        st.session_state.domanda_utente = ""

    # Creiamo due colonne: 4 parti per l'input, 1 parte per il bottone
    col_input, col_btn = st.columns([4, 1])

    with col_input:
        st.text_input("Chiedi al chatbot:", key="domanda_utente", on_change=invia, label_visibility="collapsed")

    with col_btn:
        st.button("Invia", on_click=invia)

    domanda_utente = st.session_state.get("domanda_inviata", "")

    # --------------------------------------------------

    # Generazione della risposta in una chain di eventi
    # domanda -> embedding -> similarity search -> risultati all'LLM -> risposta

    def formatta_documento(documenti):
        return "\n\n".join([documento.page_content for documento in documenti])
    
    # Quando userò il prompt, qui dentro dovrà essere inserito qualcosa chiamato "context"
    # e qualcosa chiamata "question"
    # Qui è come nei roles di ChatGPT, ma qui siamo in Langchain
    # e la struttura è più semplice: "system" e "human"
    # Attenzione che nelle stringhe ''' vengono conservati spazi e indentazioni!
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         '''Sei Massimo, assistente virtuale di ERSU Palermo.

Il tuo compito è aiutare gli studenti universitari fornendo risposte precise, chiare e affidabili basate sul contesto fornito.

Regole di comportamento:

- Rispondi sempre in modo diretto e sicuro.

- Usa un tono umano, disponibile e leggermente informale, ma professionale.

- Evita giri di parole o frasi vaghe.

- Non inventare informazioni.

- Se una risposta non è presente nel contesto o non sei sicuro, rispondi:

  "Scusami, al momento non ho informazioni precise su questo argomento."

- Non citare mai il contesto o i documenti interni.

- Dai priorità alla chiarezza pratica per lo studente.

- Quando utile, usa elenchi puntati.

- Usa emoji solo se migliorano la comunicazione 😊
    Contesto:\n{context}'''),
        ("human", "{question}")
        ])

    comparatore = vettori.as_retriever(
        # mmr = maximal marginal relevance
        search_type="mmr",
        # Ritorna i 4 frammenti più simili
        search_kwargs={"k": 4})
    
    modello_llm = ChatOpenAI(
        model="gpt-5.4-nano",
        temperature=0.3,
        max_tokens=1000,
        openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    catena = (
        # All'inizio mettiamo un dizionario che serve a costruire 
        # la struttura che il prompt vuol in input
        # Il comparatore produce i documenti (es. k=4) e li passa alla formattazione
        # RunnablePassthrough() vuol dire:
        # quando arriverà un input → passalo così com’è
        # Dobbiamo fare così perché ancora l'input concreto non c'è!  
        {"context": comparatore | formatta_documento, 
         "question": RunnablePassthrough()}
        | prompt
        | modello_llm
        | StrOutputParser()
        )
        # StrOutputParser() prende l’output del modello 
        # e lo traforma in una stringa semplice (senza aggiunta di info ecc.)
    
    if domanda_utente:
        risposta = catena.invoke(domanda_utente)
        st.write(risposta)
    st.markdown("""
<style>
    /* Aggiungiamo un padding al contenitore principale per evitare sovrapposizioni */
    .main .block-container {
        padding-bottom: 80px !important; 
    }
    
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        color: white;
        text-align: center;
        padding: 15px 0;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
        font-size: 0.9rem;
        z-index: 999;
    }
    .footer a {
        color: #EAF6FF;
        text-decoration: none;
        font-weight: bold;
    }
</style>

<div class="footer">
    <p>
        <strong>ERSU Palermo</strong> | 
        Email: <a href="mailto:info@ersupalermo.it">info@ersupalermo.it</a> | 
        Telefono: 091.6541111 | 
        <a href="https://www.ersupalermo.it/" target="_blank">Sito Ufficiale</a>
    </p>
</div>
""", unsafe_allow_html=True)
