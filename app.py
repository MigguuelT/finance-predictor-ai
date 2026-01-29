import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import google.generativeai as genai
import plotly.graph_objects as go

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(
    page_title="Preditor Financeiro RNA + Gemini 2.0",
    page_icon="📈",
    layout="wide"
)

# 2. CONFIGURAÇÃO DE SEGURANÇA (SECRETS) E MODELO
try:
    # Puxa a chave das configurações de Secrets do Streamlit
    GEMINI_CHAVE = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_CHAVE)
    
    # Utilizando a versão mais recente e rápida: Gemini 2.0 Flash
    model_gemini = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    st.error("Erro ao configurar API do Gemini. Verifique os Secrets.")
    st.stop()

# 3. INTERFACE LATERAL (SIDEBAR)
st.sidebar.header("⚙️ Parâmetros do Modelo")
ticker_input = st.sidebar.text_input("Ticker do Ativo (ex: PETR4.SA, AAPL, BTC-USD)", value="PETR4.SA").upper()
periodo_selecionado = st.sidebar.selectbox("Histórico para Treino", ["2y", "5y", "10y"], index=0)
epocas_treino = st.sidebar.slider("Épocas de Treino (RNA)", 10, 100, 20)

st.title("📈 Predição de Ativos com RNA & Gemini 2.0 Flash")
st.markdown("---")

# 4. EXECUÇÃO DO PROCESSO
if st.sidebar.button("📊 Iniciar Previsão"):
    with st.spinner(f'Buscando dados de {ticker_input} e treinando rede neural...'):
        
        # Coleta de dados
        dados = yf.download(ticker_input, period=periodo_selecionado)
        
        if not dados.empty:
            # Preparação de Dados (Min-Max Scaling)
            # 
            precos_fechamento = dados[['Close']].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            dados_normalizados = scaler.fit_transform(precos_fechamento)

            # Criando X (Hoje) e Y (Amanhã)
            X = dados_normalizados[:-1]
            y = dados_normalizados[1:]

            # Construção da Rede Neural Artificial
            # 
            rna = Sequential([
                Dense(64, activation='relu', input_dim=1),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            rna.compile(optimizer='adam', loss='mean_squared_error')
            
            # Treino silencioso
            rna.fit(X, y, epochs=epocas_treino, verbose=0)

            # Predição para o próximo dia
            ultimo_preco_norm = dados_normalizados[-1].reshape(1, 1)
            predicao_norm = rna.predict(ultimo_preco_norm)
            preco_previsto = scaler.inverse_transform(predicao_norm)[0][0]
            preco_atual = precos_fechamento[-1][0]

            # 5. EXIBIÇÃO DOS RESULTADOS
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader(f"Movimentação Histórica: {ticker_input}")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dados.index, y=dados['Close'], name='Preço de Fechamento', line=dict(color='#00ffcc')))
                fig.update_layout(template="plotly_dark", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Resultados da RNA")
                variacao = ((preco_previsto / preco_atual) - 1) * 100
                
                st.metric("Preço Atual", f"R$ {preco_atual:.2f}")
                st.metric("Previsão (Próx. Fechamento)", f"R$ {preco_previsto:.2f}", delta=f"{variacao:.2f}%")
                
                st.write("---")
                st.subheader("🤖 Insight Gemini 2.0 Flash")
                
                # Prompt otimizado para análise técnica
                prompt = (f"Analise o ativo {ticker_input}. Preço atual: {preco_atual:.2f}. "
                          f"Nossa RNA previu uma variação de {variacao:.2f}% para o próximo fechamento. "
                          f"Dê um resumo técnico curto (máximo 4 linhas) sobre possíveis suportes ou resistências "
                          f"baseado no contexto atual de mercado para este ticker.")
                
                try:
                    response = model_gemini.generate_content(prompt)
                    st.info(response.text)
                except Exception as e:
                    st.warning("Não foi possível gerar o insight do Gemini no momento.")

        else:
            st.error("Não encontramos dados para o Ticker informado. Verifique se ele existe no Yahoo Finance.")

# Rodapé
st.markdown("---")
st.caption("Aviso: As previsões são baseadas em modelos matemáticos e não constituem recomendação de investimento.")