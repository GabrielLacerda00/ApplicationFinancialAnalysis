import datetime
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


st.set_page_config(
    page_title="Inteligência Financeira",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


COR_PRIMARIA = "#00C2A8"
COR_SECUNDARIA = "#FF4B4B"
COR_NEUTRA = "#8B949E"


st.markdown("""
    <style>
        div[data-testid="stMetric"] {
            background-color: #161B22;
            border: 1px solid #21262D;
            border-radius: 10px;
            padding: 14px 16px;
        }
        div[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #8B949E; }
        .block-container { padding-top: 2rem; }
        h1 { padding-bottom: 0.2rem; }
    </style>
""", unsafe_allow_html=True)

TICKERS = {
    "APPLE": "AAPL", "GOOGLE": "GOOG", "MICROSOFT": "MSFT",
    "TESLA": "TSLA", "NETFLIX": "NFLX", "META": "META",
}


st.sidebar.markdown("## 📈 Inteligência Financeira")
st.sidebar.caption("Previsão de preços de ativos em tempo real")
st.sidebar.divider()

empresa = st.sidebar.selectbox("Empresa", list(TICKERS.keys()))
stock = TICKERS[empresa]

num_pontos_dados = st.sidebar.number_input(
    "Registros para treinar o modelo", value=365, min_value=30, step=1
)

today = datetime.date.today()
data_padrao = today - datetime.timedelta(days=int(num_pontos_dados))

col_a, col_b = st.sidebar.columns(2)
start_date = col_a.date_input("Início", value=data_padrao)
end_date = col_b.date_input("Fim", value=today)

st.sidebar.divider()
pagina = st.sidebar.radio(
    "Navegação",
    ["📊 Gráficos", "🗂️ Tabela de Dados", "🔮 Previsões"],
    label_visibility="collapsed",
)


@st.cache_data(ttl=3600, show_spinner="Baixando dados do ativo...")
def download_dados(ticker, start, end):
    ativo = yf.Ticker(ticker)
    df = ativo.history(start=start, end=end)
    return df


df_dados = download_dados(stock, start_date, end_date)

if df_dados.empty:
    st.error(
        f"Nenhum dado encontrado para **{stock}** no período selecionado. "
        "Ajuste as datas na barra lateral."
    )
    st.stop()


st.title("Inteligência Financeira")
st.caption(f"{empresa} · {stock} · {start_date} a {end_date}")

ultimo_close = df_dados["Close"].iloc[-1]
penultimo_close = df_dados["Close"].iloc[-2] if len(df_dados) > 1 else ultimo_close
variacao = ultimo_close - penultimo_close
variacao_pct = (variacao / penultimo_close * 100) if penultimo_close else 0
volume = df_dados["Volume"].iloc[-1]
maxima_periodo = df_dados["Close"].max()
minima_periodo = df_dados["Close"].min()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Último Fechamento", f"${ultimo_close:,.2f}", f"{variacao:+.2f} ({variacao_pct:+.2f}%)")
c2.metric("Volume (último dia)", f"{volume:,.0f}")
c3.metric("Máxima no Período", f"${maxima_periodo:,.2f}")
c4.metric("Mínima no Período", f"${minima_periodo:,.2f}")

st.divider()


def grafico_linha(series_dict, titulo):
    fig = go.Figure()
    cores = [COR_PRIMARIA, COR_SECUNDARIA, COR_NEUTRA, "#F2C744"]
    for i, (nome, serie) in enumerate(series_dict.items()):
        fig.add_trace(go.Scatter(
            x=serie.index, y=serie.values, mode="lines",
            name=nome, line=dict(color=cores[i % len(cores)], width=2),
        ))
    fig.update_layout(
        title=titulo, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=60, b=10), height=450,
    )
    st.plotly_chart(fig, width='stretch')



def indicadores_tecnicos():
    st.subheader("Indicadores Técnicos")

    indicador = st.radio(
        "Selecione um indicador",
        ["Preço de Fechamento", "Bandas de Bollinger", "MACD",
         "RSI", "Média Móvel Simples", "Média Móvel Exponencial"],
        horizontal=True,
    )

    close = df_dados["Close"]

    if indicador == "Preço de Fechamento":
        grafico_linha({"Fechamento": close}, "Preço de Fechamento")

    elif indicador == "Bandas de Bollinger":
        bb = BollingerBands(close)
        grafico_linha({
            "Fechamento": close,
            "Banda Superior": bb.bollinger_hband(),
            "Banda Inferior": bb.bollinger_lband(),
        }, "Bandas de Bollinger")

    elif indicador == "MACD":
        macd_obj = MACD(close)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=close.index, y=macd_obj.macd(), name="MACD", line=dict(color=COR_PRIMARIA)))
        fig.add_trace(go.Scatter(x=close.index, y=macd_obj.macd_signal(), name="Sinal", line=dict(color=COR_SECUNDARIA)))
        fig.add_trace(go.Bar(x=close.index, y=macd_obj.macd_diff(), name="Histograma", marker_color=COR_NEUTRA))
        fig.update_layout(
            title="Moving Average Convergence Divergence", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=10, r=10, t=60, b=10), height=450,
        )
        st.plotly_chart(fig, width='stretch')

    elif indicador == "RSI":
        rsi = RSIIndicator(close).rsi()
        grafico_linha({"RSI": rsi}, "Relative Strength Index")
        st.caption("Acima de 70 costuma indicar sobrecompra; abaixo de 30, sobrevenda.")

    elif indicador == "Média Móvel Simples":
        sma = SMAIndicator(close, window=14).sma_indicator()
        grafico_linha({"Fechamento": close, "SMA (14)": sma}, "Simple Moving Average")

    else:
        ema = EMAIndicator(close).ema_indicator()
        grafico_linha({"Fechamento": close, "EMA": ema}, "Exponential Moving Average")


def imprime_tabela():
    st.subheader("Tabela de Dados")
    st.caption("Últimos 20 registros do período selecionado")
    st.dataframe(
        df_dados.tail(20).style.format(precision=2),
        width='stretch',
    )

def gerando_previsoes(num):
    df = df_dados[["Close"]].copy()
    df["target"] = df["Close"].shift(-num)

    x = df.drop(["target"], axis=1).values
    y = df["target"].values[:-num]

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x[:-num], y, test_size=0.2, random_state=42
    )

    padronizador = StandardScaler()
    x_treino_padronizado = padronizador.fit_transform(x_treino)
    x_teste_padronizado = padronizador.transform(x_teste)

    modelo = LinearRegression().fit(x_treino_padronizado, y_treino)
    previsoes_teste = modelo.predict(x_teste_padronizado)
    acuracia = r2_score(y_teste, previsoes_teste)

    x_forecast = df.drop(["target"], axis=1).values[-num:]
    x_forecast_padronizado = padronizador.transform(x_forecast)
    forecast = modelo.predict(x_forecast_padronizado)

    st.metric("Acurácia do Modelo (R²)", f"{acuracia:.3f}")

    datas_futuras = pd.bdate_range(start=df_dados.index[-1], periods=num + 1)[1:]
    df_forecast = pd.DataFrame({"Data": datas_futuras, "Previsão": forecast})

    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.dataframe(
            df_forecast.set_index("Data").style.format({"Previsão": "${:,.2f}"}),
            width='stretch',
        )
    with col2:
        historico = df_dados["Close"].tail(30)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=historico.index, y=historico.values, name="Histórico", line=dict(color=COR_NEUTRA)))
        fig.add_trace(go.Scatter(x=df_forecast["Data"], y=df_forecast["Previsão"], name="Previsão",
                                  line=dict(color=COR_PRIMARIA, dash="dash"), mode="lines+markers"))
        fig.update_layout(
            title="Histórico x Previsão", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=10, r=10, t=60, b=10), height=380,
        )
        st.plotly_chart(fig, width='stretch')


def previsoes():
    st.subheader("Previsões")
    num = st.number_input("Prever quantos dias à frente?", value=1, min_value=1, step=1)
    if st.button("Gerar Previsão", type="primary"):
        with st.spinner("Treinando modelo..."):
            gerando_previsoes(int(num))



if pagina == "📊 Gráficos":
    indicadores_tecnicos()
elif pagina == "🗂️ Tabela de Dados":
    imprime_tabela()
else:
    previsoes()