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

/* SFONDO GENERALE */

.stApp {

    background: linear-gradient(

        135deg,

        #07121D,

        #0B1F2E,

        #102C3D

    );

    color: #D9F6FF;

}

/* TITOLO */

h1, h2, h3 {

    color: #7BE7FF;

    font-weight: 700;

}

/* LASTRA CENTRALE */

.main .block-container {

    background: rgba(20, 35, 50, 0.72);

    border: 1px solid rgba(123, 231, 255, 0.25);

    backdrop-filter: blur(18px);

    border-radius: 28px;

    padding: 2rem;

    margin-top: 2rem;

    box-shadow:

        0 0 25px rgba(123, 231, 255, 0.10),

        0 0 80px rgba(0, 183, 255, 0.08);

}

/* INPUT CHAT */

.stTextInput input {

    background-color: rgba(10, 25, 40, 0.9);

    color: #D9F6FF;

    border: 1px solid #46D9FF;

    border-radius: 14px;

    padding: 12px;

    font-size: 16px;

}

/* FOCUS INPUT */

.stTextInput input:focus {

    border: 1px solid #7BE7FF;

    box-shadow: 0 0 12px rgba(123, 231, 255, 0.45);

}

/* BOTTONI */

.stButton button {

    background: linear-gradient(

        90deg,

        #46D9FF,

        #00B7FF

    );

    color: white;

    border: none;

    border-radius: 14px;

    padding: 10px 20px;

    font-weight: bold;

    transition: 0.3s;

}

/* HOVER */

.stButton button:hover {

    transform: scale(1.03);

    box-shadow: 0 0 18px rgba(70, 217, 255, 0.55);

}

/* RISPOSTA CHAT */

.stMarkdown {

    color: #DFFBFF;

}

/* SIDEBAR */

section[data-testid="stSidebar"] {

    background-color: #08141F;

    border-right: 1px solid rgba(123, 231, 255, 0.15);

}

</style>

""",
unsafe_allow_html=True)

st.header("Assistenza online")

st.image("Chatbot (1).webp", width=500)

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

    st.text_input("Chiedi al chatbot:", key="domanda_utente", on_change=invia)
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
         '''Sei un assistente virtuale. 
    Usa il contesto fornito per rispondere alla domanda in modo conciso. 
    Puoi accedere a informazioni esterne, come Internet. 
    Se non conosci la risposta, dì semplicemente 'Non sono in grado di rispondere'. 
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
