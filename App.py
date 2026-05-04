import streamlit as st
import pandas as pd
#import ydata_profiling # <-- Apenas a versão moderna
#from streamlit_pandas_profiling import st_profile_report
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
#import streamlit.components.v1 as components

# Configuração da página
st.set_page_config(page_title="Auto-ML Agent Híbrido", layout="wide")
st.title("🤖 Orquestrador de Data Science")

# Sidebar
st.sidebar.header("Configurações")
api_key = st.sidebar.text_input("Insira sua Groq API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Suba seu dataset (CSV)", type="csv")

if st.sidebar.button("🧹 Limpar Histórico do Chat"):
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibição do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- FUNÇÃO DE CACHE PARA O RELATÓRIO NÃO TRAVAR O CHAT ---
#@st.cache_data
#def gerar_relatorio(dataframe):
#    return ydata_profiling.ProfileReport(dataframe, explorative=True)
# -----------------------------------------------------------

if uploaded_file:
    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(uploaded_file)
    df = st.session_state.df

    # Botão de pré-processamento
    if st.sidebar.button("🔧 Pré-processar Dados"):
        with st.spinner("Aplicando limpeza automática..."):
            df = df.drop_duplicates()
            
            # Preencher nulos
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna("missing")
            
            # NOTA: Normalização Z-score foi removida para não descaracterizar 
            # os valores reais (ex: Preço do Aluguel) durante o chat.
            
            st.session_state.df = df
            st.cache_data.clear() # Limpa o cache para recriar o relatório com os dados limpos
            st.sidebar.success("Pré-processamento concluído!")
            st.rerun()

    # Pré-análise automática local (Usando Cache)
    # Pré-análise automática local (Segura contra erros de Node)
    #with st.expander("📊 Abrir Relatório Automático (YData Profiling)", expanded=False):
    #    st.write("Relatório gerado automaticamente:")
    #    profile = gerar_relatorio(df)
        
        # Transforma o relatório em HTML puro e isola na tela
        export_html = profile.to_html()
        components.html(export_html, height=800, scrolling=True)

    # Visualização rápida
    with st.expander("👀 Visualizar Dados", expanded=False):
        st.dataframe(df.head())

   # Agente LLM com Groq (Velocidade e Estabilidade)
    if api_key:
        try:
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                groq_api_key=api_key,
                temperature=0
            )

            agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                agent_type="tool-calling", # A mágica que resolve tudo de primeira
                allow_dangerous_code=True,
                handle_parsing_errors=True,
                max_iterations=3,
                number_of_head_rows=2
            )

            if prompt := st.chat_input("O que vamos analisar hoje?"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Analisando os dados com a velocidade da Groq..."):
                        try:
                            # Chama o agente de forma direta e segura
                            response = agent.invoke({"input": prompt})
                            st.markdown(response["output"])
                            st.session_state.messages.append({"role": "assistant", "content": response["output"]})
                        except Exception as e:
                            st.error(f"O agente encontrou uma dificuldade: {str(e)}\n\nTente fazer a pergunta de uma forma mais direta.")

        except Exception as e:
            st.error(f"Erro na configuração do Agente: {str(e)}")
