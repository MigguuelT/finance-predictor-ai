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

# 1. INICIALIZAÇÃO E CONFIGURAÇÃO
st.set_page_config(page_title="IA Financeira Pro: Auditoria & Insight", layout="wide")

# Função para garantir que a API seja configurada sem erros de primeira chamada
@st.cache_resource
def configurar_ai():
    try:
        chave = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=chave)
        return genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        st.error(f"Erro ao configurar API: {e}")
        return None

model_gemini = configurar_ai()

# 2. FUNÇÕES TÉCNICAS
def calcular_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 3. INTERFACE LATERAL
st.sidebar.header("⚙️ Painel de Controle")
ticker = st.sidebar.text_input("Ticker (Ex: IAU, PETR4.SA, AAPL)", value="IAU").upper()
periodo = st.sidebar.selectbox("Histórico de Análise", ["2y", "5y", "10y"], index=0)

st.title("💹 Sistema de Inteligência Financeira")
st.caption("Análise de Tendência, Previsão por Rede Neural e Auditoria de Assertividade")

# 4. BOTÃO DE EXECUÇÃO COM TRATAMENTO DE ERRO
if st.sidebar.button("🚀 Iniciar Relatório Completo"):
    try:
        with st.spinner('Acessando dados e treinando IA...'):
            df = yf.download(ticker, period=periodo)
            
            if df.empty:
                st.error("Não foi possível encontrar dados para este Ticker.")
            else:
                # Limpeza de MultiIndex para tickers americanos
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Indicadores Técnicos
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['MA50'] = df['Close'].rolling(window=50).mean()
                df['RSI'] = calcular_rsi(df['Close'])
                
                # Criando as Abas
                aba_prev, aba_auditoria = st.tabs(["🔮 Previsão & Analise", "⚖️ Auditoria de Performance"])

                # --- ABA 1: PREVISÃO E INSIGHTS DETALHADOS ---
                with aba_prev:
                    col_p1, col_p2 = st.columns([2, 1])
                    
                    # Treino da RNA para o Futuro
                    dados_v = df[['Close']].dropna().values
                    scaler = MinMaxScaler()
                    norm = scaler.fit_transform(dados_v)
                    X_f, y_f = norm[:-1], norm[1:]
                    
                    model_f = Sequential([Dense(64, activation='relu', input_dim=1), Dense(32, activation='relu'), Dense(1)])
                    model_f.compile(optimizer='adam', loss='mse')
                    model_f.fit(X_f, y_f, epochs=30, verbose=0)
                    
                    pred_f = scaler.inverse_transform(model_f.predict(norm[-1].reshape(1,1)))[0][0]
                    preco_atual = dados_v[-1][0]
                    rsi_atual = df['RSI'].iloc[-1]

                    with col_p1:
                        fig_temp = go.Figure()
                        fig_temp.add_trace(go.Scatter(x=df.index, y=df['Close'].values.flatten(), name='Preço', line=dict(color='#00d4ff', width=2)))
                        fig_temp.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='Média 20d', line=dict(dash='dash', color='#ffcc00')))
                        fig_temp.update_layout(template="plotly_dark", title=f"Evolução Temporal: {ticker}", hovermode="x unified")
                        st.plotly_chart(fig_temp, use_container_width=True)

                    with col_p2:
                        st.metric("Preço Atual", f"{preco_atual:.2f}")
                        st.metric("Previsão RNA", f"{pred_f:.2f}", delta=f"{((pred_f/preco_atual)-1)*100:.2f}%")
                        st.write(f"**RSI (14d):** {rsi_atual:.2f}")
                        st.markdown("---")
                        
                        # INSIGHT DETALHADO DO GEMINI
                        st.subheader("🤖 Analise do Especialista")
                        prompt_detalhado = (
                            f"Como analista financeiro senior, examine o ativo {ticker}. "
                            f"Preço atual: {preco_atual:.2f}. Previsão da RNA para amanhã: {pred_f:.2f}. "
                            f"O RSI está em {rsi_atual:.2f} (sobrecompra > 70, sobrevenda < 30). "
                            f"A Média Móvel de 20 dias está em {df['MA20'].iloc[-1]:.2f}. "
                            f"Forneça uma análise técnica detalhada sobre a força da tendência e os níveis de suporte/resistência."
                        )
                        if model_gemini:
                            res = model_gemini.generate_content(prompt_detalhado)
                            st.info(res.text)

                # --- ABA 2: AUDITORIA (BACKTESTING) ---
                with aba_auditoria:
                    preco_real_ontem = float(df['Close'].iloc[-1])
                    preco_anteontem = float(df['Close'].iloc[-2])
                    
                    # Treino Cego
                    dados_aud = df['Close'].iloc[:-1].values.reshape(-1, 1)
                    scaler_aud = MinMaxScaler()
                    norm_aud = scaler_aud.fit_transform(dados_aud)
                    model_aud = Sequential([Dense(32, activation='relu', input_dim=1), Dense(1)])
                    model_aud.compile(optimizer='adam', loss='mse')
                    model_aud.fit(norm_aud[:-1], norm_aud[1:], epochs=30, verbose=0)
                    
                    pred_ontem = float(scaler_aud.inverse_transform(model_aud.predict(scaler_aud.transform([[preco_anteontem]])))[0][0])
                    acertou = (preco_real_ontem > preco_anteontem) == (pred_ontem > preco_anteontem)
                    
                    st.metric("Assertividade de Direção", "CORRETA ✅" if acertou else "INCORRETA ❌")
                    
                    fig_bar = go.Figure(data=[
                        go.Bar(name='Fechamento Real', x=['Ontem'], y=[preco_real_ontem], marker_color='#00d4ff', width=0.2),
                        go.Bar(name='Previsão da IA', x=['Ontem'], y=[pred_ontem], marker_color='#ffcc00', width=0.2)
                    ])
                    # Linhas de Margem de Erro
                    fig_bar.add_hline(y=pred_ontem*1.01, line_dash="dot", line_color="gray", annotation_text="+1%")
                    fig_bar.add_hline(y=pred_ontem*0.99, line_dash="dot", line_color="gray", annotation_text="-1%")
                    
                    fig_bar.update_layout(template="plotly_dark", barmode='group', title="Auditoria: Real vs Previsto", height=450)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    st.caption(f"Desvio de preço na auditoria: {abs((pred_ontem-preco_real_ontem)/preco_real_ontem)*100:.2f}%")

    except Exception as e:
        st.error(f"Ocorreu um erro durante o processamento: {e}")
        st.info("Dica: Clique no botão novamente para reconfirmar a conexão com os servidores.")

st.markdown("---")
st.caption("Fins educacionais. O mercado financeiro possui riscos e as predições desta ferramenta não constituem recomendação de compra ou venda.")