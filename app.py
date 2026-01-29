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

# 1. CONFIGURAÇÃO E IA
st.set_page_config(page_title="IA Financeira: Previsão & Auditoria", layout="wide")

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model_gemini = genai.GenerativeModel('gemini-2.0-flash')
except:
    st.error("Erro na API Key nos Secrets.")
    st.stop()

# 2. FUNÇÕES TÉCNICAS
def calcular_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    return 100 - (100 / (1 + (gain / loss)))

# 3. INTERFACE LATERAL
st.sidebar.header("⚙️ Configurações")
ticker = st.sidebar.text_input("Ticker (Ex: IAU, PETR4.SA, AAPL)", value="IAU").upper()
periodo = st.sidebar.selectbox("Histórico", ["2y", "5y", "10y"])

st.title("💹 Inteligência Financeira: Previsão vs Auditoria")

# 4. PROCESSAMENTO DE DADOS
if st.sidebar.button("🚀 Iniciar Análise Completa"):
    with st.spinner('Processando dados e treinando Redes Neurais...'):
        df = yf.download(ticker, period=periodo)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            # Cálculo de Médias e RSI
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['RSI'] = calcular_rsi(df['Close'])
            
            # --- CRIAÇÃO DAS ABAS ---
            aba_prev, aba_auditoria = st.tabs(["🔮 Previsão de Futuro", "⚖️ Auditoria de Acertos"])

            # --- ABA 1: PREVISÃO ---
            with aba_prev:
                col_p1, col_p2 = st.columns([2, 1])
                with col_p1:
                    fig_temp = go.Figure()
                    fig_temp.add_trace(go.Scatter(x=df.index, y=df['Close'].values.flatten(), name='Preço', line=dict(color='#00d4ff')))
                    fig_temp.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='Média 20d', line=dict(dash='dash', color='#ffcc00')))
                    fig_temp.update_layout(template="plotly_dark", title=f"Evolução Temporal: {ticker}", hovermode="x unified")
                    st.plotly_chart(fig_temp, use_container_width=True)
                
                with col_p2:
                    # Treino para Futuro
                    dados_f = df[['Close']].dropna().values
                    scaler = MinMaxScaler()
                    norm_f = scaler.fit_transform(dados_f)
                    model_f = Sequential([Dense(64, activation='relu', input_dim=1), Dense(1)])
                    model_f.compile(optimizer='adam', loss='mse')
                    model_f.fit(norm_f[:-1], norm_f[1:], epochs=30, verbose=0)
                    
                    pred_f = scaler.inverse_transform(model_f.predict(norm_f[-1].reshape(1,1)))[0][0]
                    preco_atual = dados_f[-1][0]
                    
                    st.metric("Preço Atual", f"{preco_atual:.2f}")
                    st.metric("Previsão Próximo Fechamento", f"{pred_f:.2f}", delta=f"{((pred_f/preco_atual)-1)*100:.2f}%")
                    st.write(f"**RSI:** {df['RSI'].iloc[-1]:.2f}")
                    st.markdown("---")
                    st.subheader("🤖 Insight Gemini")
                    st.info(model_gemini.generate_content(f"Analise o ativo {ticker} com preço {preco_atual:.2f} e previsão de {pred_f:.2f}. Curto e grosso.").text)

            # --- ABA 2: AUDITORIA ---
            with aba_auditoria:
                # Simulação de Ontem
                preco_real_ontem = float(df['Close'].iloc[-1])
                preco_anteontem = float(df['Close'].iloc[-2])
                
                # Treino "Cego" (sem o dado de ontem)
                dados_aud = df['Close'].iloc[:-1].values.reshape(-1, 1)
                scaler_aud = MinMaxScaler()
                norm_aud = scaler_aud.fit_transform(dados_aud)
                model_aud = Sequential([Dense(32, activation='relu', input_dim=1), Dense(1)])
                model_aud.compile(optimizer='adam', loss='mse')
                model_aud.fit(norm_aud[:-1], norm_aud[1:], epochs=30, verbose=0)
                
                pred_ontem = float(scaler_aud.inverse_transform(model_aud.predict(scaler_aud.transform([[preco_anteontem]])))[0][0])
                acertou = (preco_real_ontem > preco_anteontem) == (pred_ontem > preco_anteontem)
                
                c_aud1, c_aud2 = st.columns([1, 1])
                with c_aud1:
                    st.metric("Direção de Ontem", "ACERTOU ✅" if acertou else "ERROU ❌")
                    # Gráfico de Barras Elegante
                    fig_bar = go.Figure(data=[
                        go.Bar(name='Real', x=['Ontem'], y=[preco_real_ontem], marker_color='#00d4ff', width=0.3),
                        go.Bar(name='Previsto', x=['Ontem'], y=[pred_ontem], marker_color='#ffcc00', width=0.3)
                    ])
                    # Adicionando Linha de Margem de Erro (2%)
                    margem_sup = pred_ontem * 1.02
                    margem_inf = pred_ontem * 0.98
                    fig_bar.add_hline(y=margem_sup, line_dash="dot", line_color="gray", annotation_text="Margem +2%")
                    fig_bar.add_hline(y=margem_inf, line_dash="dot", line_color="gray", annotation_text="Margem -2%")
                    
                    fig_bar.update_layout(template="plotly_dark", barmode='group', height=400, title="Validação de Preço (Real vs Previsto)")
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with c_aud2:
                    erro_p = abs((pred_ontem - preco_real_ontem)/preco_real_ontem)*100
                    st.write(f"**Desvio da RNA:** {erro_p:.2f}%")
                    st.write("A auditoria serve para validar se a IA está conseguindo ler o 'sentimento' do mercado nas últimas 48h.")
                    st.markdown("---")
                    st.write("📥 **Exportação:**")
                    csv = pd.DataFrame({"Ticker":[ticker],"Real":[preco_real_ontem],"Previsto":[pred_ontem],"Acerto":[acertou]}).to_csv(index=False).encode('utf-8')
                    st.download_button("Baixar Auditoria CSV", csv, f"auditoria_{ticker}.csv")

        else:
            st.error("Erro ao carregar dados.")

# Rodapé
st.markdown("---")
st.caption("Desenvolvido para análise de Ativos e ETFs (IAU, SLV, TFLO, NUKZ).")