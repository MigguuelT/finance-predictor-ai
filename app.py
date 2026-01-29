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
st.set_page_config(page_title="IA Financeira Pro: Evolução Temporal & RNA", page_icon="📉", layout="wide")

# 2. CONFIGURAÇÃO DE SEGURANÇA E MODELO (GEMINI 2.0 FLASH)
try:
    GEMINI_CHAVE = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_CHAVE)
    model_gemini = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    st.error("Erro ao configurar API do Gemini. Verifique os Secrets no Streamlit Cloud.")
    st.stop()

# 3. FUNÇÃO PARA INDICADOR RSI (Índice de Força Relativa)
def calcular_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 4. INTERFACE LATERAL
st.sidebar.header("🔍 Painel de Controle")
ticker_input = st.sidebar.text_input("Ticker (Ex: PETR4.SA, IAU, SLV, NUK, AAPL)", value="IAU").upper()
periodo = st.sidebar.selectbox("Período de Análise Temporal", ["1y", "2y", "5y", "10y"], index=1)
epocas = st.sidebar.slider("Refinamento da RNA (Épocas)", 10, 100, 30)

st.title("📈 Evolução Temporal e Predição com IA")
st.markdown(f"Análise avançada do ativo: **{ticker_input}**")

# 5. PROCESSAMENTO
if st.sidebar.button("Executar Análise Completa"):
    with st.spinner('Baixando histórico, calculando indicadores e treinando RNA...'):
        # Coleta de dados via Yahoo Finance
        dados = yf.download(ticker_input, period=periodo)
        
        if not dados.empty:
            # Cálculo de Médias Móveis e RSI
            dados['MA20'] = dados['Close'].rolling(window=20).mean()
            dados['MA50'] = dados['Close'].rolling(window=50).mean()
            dados['RSI'] = calcular_rsi(dados['Close'])
            
            # --- GRÁFICO DE EVOLUÇÃO TEMPORAL INTERATIVO ---
            st.subheader(f"📊 Evolução Temporal: {ticker_input}")
            fig = go.Figure()
            
            # Preço de Fechamento
            fig.add_trace(go.Scatter(x=dados.index, y=dados['Close'], name='Preço de Fechamento', 
                                     line=dict(color='#00d4ff', width=2)))
            # Média Móvel Curta
            fig.add_trace(go.Scatter(x=dados.index, y=dados['MA20'], name='Média Móvel 20d', 
                                     line=dict(color='#ffcc00', width=1.5, dash='dash')))
            # Média Móvel Longa
            fig.add_trace(go.Scatter(x=dados.index, y=dados['MA50'], name='Média Móvel 50d', 
                                     line=dict(color='#ff3300', width=1.5, dash='dot')))
            
            fig.update_layout(
                template="plotly_dark",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_title="Tempo",
                yaxis_title="Preço (Moeda Original)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- PARTE DA REDE NEURAL ---
            # Limpeza de dados para treino (remove NaNs gerados pelos indicadores)
            dados_limpos = dados[['Close']].dropna()
            precos_v = dados_limpos.values
            scaler = MinMaxScaler(feature_range=(0, 1))
            dados_norm = scaler.fit_transform(precos_v)
            
            X, y = dados_norm[:-1], dados_norm[1:]
            
            # Arquitetura da RNA
            rna = Sequential([
                Dense(64, activation='relu', input_dim=1),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            rna.compile(optimizer='adam', loss='mse')
            rna.fit(X, y, epochs=epocas, verbose=0)

            # Predição do próximo ponto
            ultimo_p = dados_norm[-1].reshape(1, 1)
            pred_n = rna.predict(ultimo_p)
            preco_previsto = scaler.inverse_transform(pred_n)[0][0]
            preco_atual = precos_v[-1][0]
            rsi_atual = dados['RSI'].iloc[-1]

            # --- EXIBIÇÃO DE MÉTRICAS E INSIGHTS ---
            st.markdown("---")
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.subheader("🎯 Resultado RNA")
                delta_p = ((preco_previsto / preco_atual) - 1) * 100
                st.metric("Preço Atual", f"{preco_atual:.2f}")
                st.metric("Previsão (Prox. Fechamento)", f"{preco_previsto:.2f}", delta=f"{delta_p:.2f}%")
                st.write(f"**RSI Atual:** {rsi_atual:.2f}")

            with c2:
                st.subheader("🤖 Parecer Técnico - Gemini 2.0 Flash")
                
                # Contexto para a IA
                prompt = (f"Analise o ativo {ticker_input}. Preço atual: {preco_atual:.2f}. "
                          f"Indicadores: RSI em {rsi_atual:.2f}, Média Móvel 20d em {dados['MA20'].iloc[-1]:.2f}. "
                          f"Nossa Rede Neural previu uma variação de {delta_p:.2f}% para o próximo fechamento. "
                          f"Como especialista financeiro, interprete se o ativo está em tendência de alta, baixa ou neutra.")
                
                try:
                    insight = model_gemini.generate_content(prompt)
                    st.info(insight.text)
                except:
                    st.warning("IA temporariamente indisponível para gerar o insight.")

        else:
            st.error("Não foi possível carregar dados para este Ticker. Verifique se ele é válido no Yahoo Finance.")

st.sidebar.markdown("---")
st.sidebar.caption("Desenvolvido para análise de ativos B3 e Globais.")