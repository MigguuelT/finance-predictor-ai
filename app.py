import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import google.generativeai as genai
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. CONFIGURAÇÃO
st.set_page_config(page_title="IA Financeira: Auditoria de Acertos", layout="wide")

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model_gemini = genai.GenerativeModel('gemini-2.0-flash')
except:
    st.error("Erro na API Key.")
    st.stop()

# 2. INTERFACE
st.title("⚖️ Auditoria de Inteligência Artificial")
st.sidebar.header("Configurações de Auditoria")
ticker = st.sidebar.text_input("Ativo para Teste", value="IAU").upper()

# 3. FUNÇÃO DE TREINO E TESTE
def calcular_acuracia(ticker_simbolo):
    # Puxamos um pouco mais de dados para garantir o treino
    df = yf.download(ticker_simbolo, period="2y")
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

    # Pegamos o preço de ONTEM (Real) e o preço de ANTEONTEM (Base para a previsão)
    preco_real_ontem = float(df['Close'].iloc[-1])
    preco_anteontem = float(df['Close'].iloc[-2])
    
    # Preparamos os dados excluindo o último dia (simulando que não sabemos o futuro)
    dados_treino = df['Close'].iloc[:-1].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    dados_norm = scaler.fit_transform(dados_treino)
    
    X, y = dados_norm[:-1], dados_norm[1:]
    
    # RNA Rápida para Auditoria
    model = Sequential([Dense(32, activation='relu', input_dim=1), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, verbose=0)
    
    # Previsão para "Ontem"
    input_anteontem = scaler.transform([[preco_anteontem]])
    pred_norm = model.predict(input_anteontem)
    previsao_ontem = float(scaler.inverse_transform(pred_norm)[0][0])
    
    # Lógica de Acerto de Direção
    subiu_real = preco_real_ontem > preco_anteontem
    previu_subida = previsao_ontem > preco_anteontem
    acertou_direcao = subiu_real == previu_subida
    
    erro_percentual = abs((previsao_ontem - preco_real_ontem) / preco_real_ontem) * 100
    
    return {
        "Real": preco_real_ontem,
        "Previsto": previsao_ontem,
        "Acertou_Direcao": acertou_direcao,
        "Erro_Preco": erro_percentual,
        "Tendencia": "ALTA" if previu_subida else "BAIXA"
    }

# 4. BOTÃO DE EXECUÇÃO
if st.sidebar.button("Auditar Estratégia"):
    with st.spinner('Realizando Backtesting em tempo real...'):
        resultado = calcular_acuracia(ticker)
        
        if resultado:
            st.subheader(f"📊 Relatório de Assertividade: {ticker}")
            
            # Métricas de Performance
            m1, m2, m3 = st.columns(3)
            
            status_cor = "normal" if resultado["Acertou_Direcao"] else "inverse"
            m1.metric("Direção do Mercado", "ACERTOU ✅" if resultado["Acertou_Direcao"] else "ERROU ❌", delta_color=status_cor)
            m2.metric("Precisão do Preço (Erro %)", f"{resultado['Erro_Preco']:.2f}%")
            m3.metric("Preço Real (Ontem)", f"{resultado['Real']:.2f}")

            # Explicação do Gemini
            st.markdown("---")
            st.subheader("🤖 Análise da Auditoria pelo Gemini")
            prompt = (f"O ativo {ticker} fechou ontem a {resultado['Real']:.2f}. "
                      f"Minha rede neural previu {resultado['Previsto']:.2f} (Erro de {resultado['Erro_Preco']:.2f}%). "
                      f"A IA previu corretamente a direção? {resultado['Acertou_Direcao']}. "
                      f"Explique brevemente por que modelos de RNA podem ter essa margem de erro em ativos como {ticker}.")
            
            st.info(model_gemini.generate_content(prompt).text)
            
            # Gráfico de Comparação
            fig = go.Figure(data=[
                go.Bar(name='Preço Real', x=['Ontem'], y=[resultado['Real']], marker_color='#00d4ff'),
                go.Bar(name='Previsão RNA', x=['Ontem'], y=[resultado['Previsto']], marker_color='#ffcc00')
            ])
            fig.update_layout(template="plotly_dark", barmode='group', title="Real vs Previsto (Último Fechamento)")
            st.plotly_chart(fig)

        else:
            st.error("Erro ao processar dados.")

# Rodapé com nota sobre seus investimentos
st.sidebar.markdown("---")
st.sidebar.info(f"Dica: TFLO e SGOV possuem baixa volatilidade, o que costuma gerar acertos de preço acima de 98% nesta RNA.")