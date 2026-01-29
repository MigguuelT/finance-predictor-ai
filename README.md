# 📈 IA Financeira: Predictor & Insight

Este projeto utiliza **Redes Neurais Artificiais (RNAs)** para prever o próximo preço de fechamento de ativos financeiros e integra o **Google Gemini 2.0 Flash** para fornecer análises contextuais sobre a previsão.

## 🚀 Funcionalidades
- **Download de Dados Reais:** Integração com Yahoo Finance (`yfinance`).
- **Deep Learning:** Rede neural densa construída com TensorFlow/Keras.
- **IA Generativa:** Análise técnica automatizada via Gemini API.
- **Interface Interativa:** Desenvolvido inteiramente em Streamlit.

## 🌟 Novas Funcionalidades (v2.0)
- **Módulo de Auditoria (Backtesting):** O sistema volta 48h no tempo, treina o modelo e tenta prever o preço de ontem. Ele compara o resultado com o fechamento real para calcular a taxa de acerto de direção.
- **Integração Gemini 2.0 Flash:** Respostas instantâneas para análise técnica e explicação de desvios de preço.
- **Tracking de Performance:** Visualização da assertividade (Acerto/Erro) diretamente no painel.
- **Exportação CSV:** Possibilidade de baixar os resultados para estudos externos.
- **Suporte Global:** Compatível com ativos B3 (`.SA`) e ETFs americanos (`IAU`, `SLV`, `TFLO`, etc.).

## 🛠️ Como rodar localmente

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/SEU_USUARIO/NOME_DO_REPO.git](https://github.com/SEU_USUARIO/NOME_DO_REPO.git)
   cd NOME_DO_REPO