import retriever
import streamlit as st

from retriever import Retriever
from augmentation import Augmentation
from generation import Generation

# Configuração da página
st.set_page_config(page_title="RAG Chat", page_icon="🤖")

TALK_ID = "sandeco-chat-001"


# Título
st.title("🤖 RAG Chat Assistant")

# Inicializar histórico
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input do usuário
if query := st.chat_input("Digite sua pergunta..."):
    # Adicionar pergunta do usuário
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Processar e responder
    with st.chat_message("assistant"):
        with st.spinner("Processando..."):
            
            retriever = Retriever(collection_name="synthetic_dataset_papers")
            augmentation = Augmentation(talk_id=TALK_ID)
            generation = Generation(model="gemini-2.5-flash")
            
            # A parte "R" do RAG
            result = retriever.search(query, n_results=10, show_metadata=False)

            # A parte "A" do RAG
            prompt = augmentation.generate_prompt(query, result)
            
            # Gerar resposta
            response = generation.generate(
                prompt
            )
            
            augmentation.add_memory(response)
            
            st.markdown(response)
    
    # Adicionar resposta ao histórico
    st.session_state.messages.append({"role": "assistant", "content": response})
