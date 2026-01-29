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
    page_title="IA Financeira Pro: Evolução & Predição",
    page_icon="💹",
    layout="wide"
)

# 2. CONFIGURAÇÃO DE SEGURANÇA E IA
try:
    # Busca a chave nos Secrets do Streamlit Cloud
    GEMINI_CHAVE = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_CHAVE)
    # Modelo Gemini 2.0 Flash para respostas instantâneas
    model_gemini = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    st.error("Erro ao configurar API. Verifique se 'GEMINI_API_KEY' está nos Secrets.")
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

st.title("💹 Sistema Avançado de Predição Financeira")
st.caption(f"Analisando dados históricos e tendências para: **{ticker_input}**")

# 5. PROCESSAMENTO PRINCIPAL
if st.sidebar.button("🚀 Gerar Relatório Completo"):
    with st.spinner('Acessando mercado e treinando modelos...'):
        # Download dos dados
        df = yf.download(ticker_input, period=periodo)
        
        if not df.empty:
            # CORREÇÃO CRÍTICA: Trata MultiIndex do yfinance (comum em ETFs e ações US)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Cálculo de Indicadores Técnicos
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['RSI'] = calcular_rsi(df['Close'])
            
            # --- GRÁFICO DE EVOLUÇÃO TEMPORAL ---
            st.subheader(f"📈 Evolução Temporal e Médias Móveis")
            fig = go.Figure()

            # Preço Real
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Close'].values.flatten(),
                name='Preço Fechamento', line=dict(color='#00d4ff', width=2.5)
            ))

            # Média 20 dias (Curta)
            df_ma20 = df['MA20'].dropna()
            fig.add_trace(go.Scatter(
                x=df_ma20.index, y=df_ma20.values,
                name='Média Móvel 20d (Curto Prazo)', line=dict(color='#ffcc00', width=1.5, dash='dash')
            ))

            # Média 50 dias (Longa)
            df_ma50 = df['MA50'].dropna()
            fig.add_trace(go.Scatter(
                x=df_ma50.index, y=df_ma50.values,
                name='Média Móvel 50d (Médio Prazo)', line=dict(color='#ff3300', width=1.5, dash='dot')
            ))

            fig.update_layout(
                template="plotly_dark",
                hovermode="x unified",
                xaxis_title="Período",
                yaxis_title="Preço",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- REDE NEURAL ARTIFICIAL (RNA) ---
            # Prepara dados (remove NaNs para o treino)
            dados_treino = df[['Close']].dropna()
            valores = dados_treino.values
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            norm_valores = scaler.fit_transform(valores)
            
            X_train = norm_valores[:-1]
            y_train = norm_valores[1:]

            # Arquitetura
            rna = Sequential([
                Dense(64, activation='relu', input_dim=1),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            rna.compile(optimizer='adam', loss='mse')
            rna.fit(X_train, y_train, epochs=epocas, verbose=0)

            # Predição
            ultimo_val = norm_valores[-1].reshape(1, 1)
            pred_n = rna.predict(ultimo_val)
            preco_previsto = scaler.inverse_transform(pred_n)[0][0]
            preco_atual = valores[-1][0]
            rsi_atual = df['RSI'].iloc[-1]

            # --- MÉTRICAS E INSIGHT GEMINI ---
            st.markdown("---")
            col_m1, col_m2, col_m3 = st.columns(3)
            
            with col_m1:
                st.metric("Preço Atual", f"{preco_atual:.2f}")
            with col_m2:
                delta_perc = ((preco_previsto/preco_atual)-1)*100
                st.metric("Previsão RNA (Próx. Fechamento)", f"{preco_previsto:.2f}", delta=f"{delta_perc:.2f}%")
            with col_m3:
                st.metric("RSI (14 dias)", f"{rsi_atual:.2f}")

            st.subheader("🤖 Análise Especializada Gemini 2.0 Flash")
            
            # Contexto para a IA
            prompt = (f"Analise o ativo {ticker_input}. O preço atual é {preco_atual:.2f}. "
                      f"O RSI está em {rsi_atual:.2f} e a Média Móvel de 20 dias está em {df['MA20'].iloc[-1]:.2f}. "
                      f"Nossa Rede Neural previu uma variação de {delta_perc:.2f}% para o próximo período. "
                      f"Dê um parecer técnico sobre o momentum (alta, baixa ou neutro) e mencione se o RSI indica sobrecompra ou sobrevenda.")
            
            try:
                insight = model_gemini.generate_content(prompt)
                st.info(insight.text)
            except Exception as e:
                st.warning("O Gemini não pôde gerar a análise no momento.")

        else:
            st.error("Ticker não encontrado ou sem dados históricos. Tente outro símbolo.")

# Rodapé
st.markdown("---")
st.caption("Nota: Este app utiliza Redes Neurais para fins educacionais e não garante lucros financeiros.")