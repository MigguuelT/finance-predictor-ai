# 💹 IA Financeira Pro: Auditoria, Predição & Insights Estratégicos

Uma plataforma avançada de análise quantitativa que integra **Redes Neurais Artificiais (RNAs)** e **IA Generativa (Gemini 2.0 Flash)** para auxiliar na tomada de decisão e validação de estratégias para ativos da B3 e Globais.

## 🚀 Diferenciais da Versão 2.1
- **Estabilidade Garantida:** Implementação de `@st.cache_resource` para evitar erros de inicialização na primeira chamada da API e carregamento de modelos.
- **Insights de Analista Sênior:** O Gemini 2.0 Flash agora realiza o cruzamento de indicadores (RSI + Médias Móveis) com a predição da RNA para gerar pareceres técnicos detalhados.
- **Navegação por Abas:** Interface organizada entre **Previsão de Futuro** e **Auditoria de Performance**.
- **Módulo de Backtesting de Curto Prazo:** Auditoria automática que valida a assertividade do modelo comparando a predição de ontem com o fechamento real.
- **Suporte a Tickers Globais:** Configurado para tratar dados complexos (MultiIndex) de ETFs americanos como `IAU`, `SLV`, `TFLO`, `SGOV` e `NUKZ`.

## 🛠️ Arquitetura Técnica
O sistema utiliza uma abordagem híbrida:
1.  **Rede Neural (Keras/TensorFlow):** Responsável pelo processamento estatístico e identificação de padrões de preço.
2.  **Lógica de Backtesting:** Uma função de "auditoria cega" que isola dados do passado para testar a acurácia direcional do modelo.
3.  **LLM (Gemini 2.0):** Atua como a camada de interpretação, transformando números e gráficos em insights acionáveis.

## ⚙️ Configuração para Streamlit Cloud
1.  No painel do Streamlit Cloud, acesse **Settings > Secrets**.
2.  Adicione sua chave de API do Google conforme o formato abaixo:
    ```toml
    GEMINI_API_KEY = "SUA_CHAVE_AQUI"
    ```

## 📦 Dependências Principais
- `yfinance`: Extração de dados de mercado em tempo real.
- `tensorflow`: Construção e treino das redes neurais.
- `google-generativeai`: Integração com o modelo Gemini 2.0 Flash.
- `plotly`: Visualização interativa de gráficos temporais e de auditoria.

---
*Aviso Legal: Os resultados gerados são baseados em modelos probabilísticos e análise histórica. O mercado financeiro possui riscos e as predições desta ferramenta não constituem recomendação de compra ou venda.*