import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import google.generativeai as genai
import plotly.graph_objects as go
from datetime import datetime

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="IA Financeira Pro: Evolução & CSV", page_icon="💹", layout="wide")

# 2. CONFIGURAÇÃO DE SEGURANÇA E IA
try:
    GEMINI_CHAVE = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_CHAVE)
    model_gemini = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    st.error("Erro ao configurar API. Verifique os Secrets.")
    st.stop()

# 3. FUNÇÃO AUXILIAR: RSI
def calcular_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 4. SIDEBAR - CONTROLES
st.sidebar.header("📊 Painel de Análise")
ticker_input = st.sidebar.text_input("Ticker (Ex: PETR4.SA, IAU, SLV, AAPL, BTC-USD)", value="IAU").upper()
periodo = st.sidebar.selectbox("Histórico Temporal", ["1y", "2y", "5y", "10y"], index=1)
epocas = st.sidebar.slider("Treinamento da RNA (Épocas)", 10, 100, 30)

st.title("💹 Predição Financeira & Histórico CSV")

# 5. PROCESSAMENTO PRINCIPAL
if st.sidebar.button("🚀 Gerar Relatório Completo"):
    with st.spinner('Processando dados...'):
        df = yf.download(ticker_input, period=periodo)
        
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['RSI'] = calcular_rsi(df['Close'])
            
            # --- GRÁFICO ---
            st.subheader(f"📈 Gráfico Interativo: {ticker_input}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'].values.flatten(), name='Preço', line=dict(color='#00d4ff')))
            fig.add_trace(go.Scatter(x=df['MA20'].dropna().index, y=df['MA20'].dropna().values, name='Média 20d', line=dict(dash='dash', color='#ffcc00')))
            fig.update_layout(template="plotly_dark", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # --- REDE NEURAL ---
            dados_treino = df[['Close']].dropna()
            valores = dados_treino.values
            scaler = MinMaxScaler()
            norm_valores = scaler.fit_transform(valores)
            
            X, y = norm_valores[:-1], norm_valores[1:]
            rna = Sequential([Dense(64, activation='relu', input_dim=1), Dense(32, activation='relu'), Dense(1)])
            rna.compile(optimizer='adam', loss='mse')
            rna.fit(X, y, epochs=epocas, verbose=0)

            pred_n = rna.predict(norm_valores[-1].reshape(1, 1))
            preco_previsto = scaler.inverse_transform(pred_n)[0][0]
            preco_atual = valores[-1][0]

            # --- MÉTRICAS ---
            col1, col2 = st.columns(2)
            col1.metric("Preço Atual", f"{preco_atual:.2f}")
            col2.metric("Previsão RNA", f"{preco_previsto:.2f}", delta=f"{((preco_previsto/preco_atual)-1)*100:.2f}%")

            # --- FUNÇÃO CSV (NOVO) ---
            st.markdown("---")
            st.subheader("📂 Exportar Dados")
            
            # Criando o DataFrame de histórico para download
            dados_export = pd.DataFrame({
                "Data_Analise": [datetime.now().strftime("%Y-%m-%d %H:%M")],
                "Ticker": [ticker_input],
                "Preco_Atual": [round(float(preco_atual), 2)],
                "Previsao_RNA": [round(float(preco_previsto), 2)],
                "RSI_Momento": [round(float(df['RSI'].iloc[-1]), 2)]
            })

            csv = dados_export.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Baixar Previsão em CSV",
                data=csv,
                file_name=f"previsao_{ticker_input}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv',
            )

            # --- INSIGHT GEMINI ---
            st.subheader("🤖 Parecer Gemini 2.0 Flash")
            prompt = f"Ativo {ticker_input}, Preço {preco_atual:.2f}, Previsão RNA {preco_previsto:.2f}. Faça uma análise curta."
            st.info(model_gemini.generate_content(prompt).text)

        else:
            st.error("Ticker não encontrado.")