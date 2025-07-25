import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gumbel_r, pearson3, kstest, anderson
from fpdf import FPDF
import tempfile
import os

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA E ESTILO VISUAL (CSS)
# ==============================================================================
st.set_page_config(
    page_title="SAP-IDF | Análise Pluviométrica",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ALTERADO: TEMA VISUAL ESCURO COM CSS CUSTOMIZADO ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Cores do tema escuro */
    :root {
        --primary-color: #56B4E9; /* Azul claro para bom contraste */
        --background-color: #0F1116; /* Fundo principal escuro */
        --secondary-background-color: #1E2129; /* Fundo de cards e containers */
        --text-color: #FAFAFA; /* Texto principal claro */
        --light-text-color: #A0A0B0; /* Texto secundário (cinza claro) */
        --border-color: #3A3D46; /* Cor das bordas */
    }
            
    /* Fundo principal */
    .stApp {
        background-color: var(--background-color);
    }

    /* Headers */
    h1, h2, h3 {
        color: var(--text-color);
    }
            
    h1 {
        font-weight: 600;
    }
    
    h2 {
        font-weight: 600;
        border-bottom: 2px solid var(--border-color);
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    /* Estilo dos cards */
    .stMetric, [data-testid="stMetric"], [data-testid="stExpander"], .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        padding: 20px;
    }

    /* Remove o preenchimento excessivo dos expanders */
    [data-testid="stExpander"] .streamlit-expanderContent {
        padding-top: 20px;
    }
            
    /* Estilo das abas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 5px;
        padding: 10px 15px;
        transition: all 0.2s ease-in-out;
        border-bottom: 2px solid transparent;
        color: var(--light-text-color);
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: var(--primary-color);
        border-bottom: 2px solid var(--primary-color);
        font-weight: 600;
    }
            
    /* Botões */
    .stButton>button {
        border-radius: 8px;
        border: 2px solid var(--primary-color);
        background-color: var(--primary-color);
        color: var(--background-color);
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: transparent;
        color: var(--primary-color);
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. FUNÇÕES AUXILIARES E CACHE (sem alteração de lógica)
# ==============================================================================

@st.cache_data
def load_data(uploaded_file):
    """Lê e processa o arquivo CSV de chuvas."""
    df_raw = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8')
    df_raw.columns = [col.strip().lower() for col in df_raw.columns]

    if "precipitacao" in df_raw.columns:
        df_raw["precipitacao"] = pd.to_numeric(
            df_raw["precipitacao"].astype(str).str.replace(",", "."), errors='coerce'
        )
    else:
        raise ValueError("Coluna 'precipitacao' não encontrada.")

    if "datahora" in df_raw.columns:
        df_raw["datahora"] = pd.to_datetime(df_raw["datahora"], errors='coerce')
    elif "data" in df_raw.columns and "hora" in df_raw.columns:
        df_raw["hora"] = df_raw["hora"].astype(str).str.zfill(4)
        df_raw["datahora"] = pd.to_datetime(
            df_raw["data"].astype(str) + " " + df_raw["hora"].str[:2] + ":" + df_raw["hora"].str[2:],
            format="%Y-%m-%d %H:%M",
            errors="coerce"
        )
    else:
        raise ValueError("Colunas de data e hora não reconhecidas. Use 'datahora' ou 'data' e 'hora'.")

    df_raw = df_raw.dropna(subset=["datahora", "precipitacao"])
    df = df_raw[["datahora", "precipitacao"]].sort_values("datahora").set_index("datahora")
    return df

@st.cache_data
def calculate_annual_maxima(_df, duration):
    """Calcula as máximas anuais para uma dada duração."""
    accumulated = _df["precipitacao"].rolling(window=duration, min_periods=1).sum()
    annual_maxima = accumulated.groupby(_df.index.year).max().dropna()
    return annual_maxima

@st.cache_data
def calculate_idf_curves(series, duration, trs_np):
    """Ajusta as distribuições Gumbel e Log-Pearson III."""
    if len(series) < 5:
        return None, None, None, None
    
    mu_g, beta_g = gumbel_r.fit(series.values)
    ks_stat, ks_p = kstest(series.values, 'gumbel_r', args=(mu_g, beta_g))
    ad_result = anderson((series.values - mu_g) / beta_g, dist='gumbel')

    dados_log = np.log10(series.values[series.values > 0])
    skew = pd.Series(dados_log).skew()
    mean_log = np.mean(dados_log)
    std_log = np.std(dados_log, ddof=1)
    
    intensities_gumbel = [gumbel_r.ppf(1 - 1/tr, loc=mu_g, scale=beta_g) for tr in trs_np]
    lp3_dist = pearson3(skew, loc=mean_log, scale=std_log)
    intensities_lp3 = [10 ** lp3_dist.ppf(1-1/tr) for tr in trs_np]

    df_idf = pd.DataFrame({
        "TR (anos)": trs_np,
        f"Gumbel_{duration}h (mm)": intensities_gumbel,
        f"LP3_{duration}h (mm)": intensities_lp3,
        f"Intensidade_Gumbel_{duration}h (mm/h)": np.array(intensities_gumbel) / duration,
        f"Intensidade_LP3_{duration}h (mm/h)": np.array(intensities_lp3) / duration
    })
    
    params_gumbel = {"mu": mu_g, "beta": beta_g, "ks_p": ks_p, "ad_stat": ad_result.statistic, "ad_crit": ad_result.critical_values}
    params_lp3 = {"mean_log": mean_log, "std_log": std_log, "skew": skew}

    return df_idf, params_gumbel, params_lp3, series

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Relatório de Análise de Chuvas Intensas - Curvas IDF', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

def generate_professional_pdf(df_idf, fig_path, duration, params_gumbel, params_lp3):
    """Gera um relatório PDF com layout profissional."""
    pdf = PDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Curvas IDF para Duração de {duration} horas", ln=True, align='C')
    pdf.ln(10)

    pdf.image(fig_path, x=10, y=None, w=190)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Tabela de Precipitações e Intensidades", ln=True)
    pdf.set_font("Arial", '', 10)
    
    pdf.set_fill_color(220, 220, 220)
    col_widths = [25, 40, 40, 45, 40]
    headers = ["TR (anos)", "Gumbel (mm)", "LP3 (mm)", "Int. Gumbel (mm/h)", "Int. LP3 (mm/h)"]
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, 1, 0, 'C', 1)
    pdf.ln()

    for _, row in df_idf.iterrows():
        pdf.cell(col_widths[0], 8, f"{row.iloc[0]}", 1, 0, 'C')
        pdf.cell(col_widths[1], 8, f"{row.iloc[1]:.2f}", 1, 0, 'C')
        pdf.cell(col_widths[2], 8, f"{row.iloc[2]:.2f}", 1, 0, 'C')
        pdf.cell(col_widths[3], 8, f"{row.iloc[3]:.2f}", 1, 0, 'C')
        pdf.cell(col_widths[4], 8, f"{row.iloc[4]:.2f}", 1, 1, 'C')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Parâmetros Estatísticos e Testes de Aderência", ln=True)
    pdf.set_font("Arial", '', 10)
    
    pdf.multi_cell(0, 6, 
        f"Distribuição Gumbel:\n"
        f"  - Parâmetro de Posição (mu): {params_gumbel['mu']:.2f} mm\n"
        f"  - Parâmetro de Escala (beta): {params_gumbel['beta']:.2f} mm\n"
        f"  - Teste K-S (p-valor): {params_gumbel['ks_p']:.4f} "
        f"({'Aceito' if params_gumbel['ks_p'] > 0.05 else 'Rejeitado'} a 5% de significância)\n"
    )
    pdf.multi_cell(0, 6,
        f"Distribuição Log-Pearson III:\n"
        f"  - Média (log10): {params_lp3['mean_log']:.3f}\n"
        f"  - Desvio Padrão (log10): {params_lp3['std_log']:.3f}\n"
        f"  - Coef. de Assimetria (log10): {params_lp3['skew']:.3f}"
    )

    return pdf

# ==============================================================================
# 3. INTERFACE DO DASHBOARD
# ==============================================================================

st.title("💧 Sistema de Análise Pluviométrica (SAP-IDF)")
st.caption("Uma ferramenta para análise de séries históricas de chuva e geração de curvas Intensidade-Duração-Frequência.")
st.divider()

with st.sidebar:
    st.header("Configurações da Análise")
    uploaded_file = st.file_uploader(
        "Carregue seu arquivo CSV",
        type=["csv"],
        help="O arquivo deve conter colunas como 'datahora' e 'precipitacao'."
    )

if uploaded_file is None:
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/263/263884.png", width=150)
    with col2:
        st.header("Bem-vindo ao SAP-IDF!")
        st.markdown("""
        Para começar, por favor, **carregue seus dados de chuva horária** usando o painel à esquerda.

        **Passos:**
        1.  Clique em **'Browse files'** na barra lateral.
        2.  Selecione o seu arquivo no formato `.csv`.
        3.  Aguarde o processamento e explore as abas de análise.
        """)
    st.stop()

try:
    with st.spinner("Analisando seu arquivo... Isso pode levar alguns segundos."):
        df = load_data(uploaded_file)
except Exception as e:
    st.error(f"❌ **Erro ao processar o arquivo:** {e}")
    st.stop()

with st.sidebar:
    st.success(f"Arquivo **{uploaded_file.name}** carregado!")
    st.divider()
    anos_disponiveis = sorted(df.index.year.unique())
    ano_min_selecionado, ano_max_selecionado = st.select_slider(
        "Selecione o intervalo de anos:",
        options=anos_disponiveis,
        value=(anos_disponiveis[0], anos_disponiveis[-1])
    )

df_filtrado = df[(df.index.year >= ano_min_selecionado) & (df.index.year <= ano_max_selecionado)]

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 **Visão Geral**", 
    "📈 **Máximas Anuais**", 
    "📉 **Curvas IDF**", 
    "📄 **Relatório PDF**"
])

with tab1:
    st.header("Resumo do Período Selecionado")
    
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Período Analisado", f"{ano_min_selecionado}–{ano_max_selecionado}")
        col2.metric("Total de Registros", f"{len(df_filtrado):,}")
        col3.metric("Precipitação Total", f"{df_filtrado['precipitacao'].sum():,.1f} mm")
        col4.metric("Máximo Horário", f"{df_filtrado['precipitacao'].max():.1f} mm")

    st.subheader("Série Temporal da Precipitação")
    with st.container():
        aggregation_level = st.selectbox(
            "Resolução do gráfico:",
            options=['Diária', 'Semanal', 'Mensal', 'Horária (p/ períodos curtos)'],
            index=0
        )

        df_plot = df_filtrado['precipitacao']
        if aggregation_level == 'Diária':
            df_plot = df_plot.resample('D').sum()
        elif aggregation_level == 'Semanal':
            df_plot = df_plot.resample('W').sum()
        elif aggregation_level == 'Mensal':
            df_plot = df_plot.resample('ME').sum()
        else:
            if len(df_filtrado) > 8760: # Limite de ~1 ano
                st.warning("A visualização horária foi desativada pois o período é muito longo. Escolha outra resolução ou um intervalo menor.")
                st.stop()

        fig_serie = px.line(x=df_plot.index, y=df_plot.values, labels={'y': 'Precipitação (mm)', 'x': 'Data'})
        # ALTERADO: Template do gráfico para 'plotly_dark'
        fig_serie.update_layout(template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig_serie.update_traces(line_color= 'var(--primary-color)', line_width=1.5)
        st.plotly_chart(fig_serie, use_container_width=True)

    col_resumo, col_hist = st.columns(2)
    with col_resumo:
        st.subheader("Resumo Anual")
        resumo_anual = df_filtrado['precipitacao'].resample("YE").agg(['sum', 'max']).rename(
            columns={'sum': 'Total (mm)', 'max': 'Máx. Horário (mm)'}
        )
        resumo_anual.index.name = "Ano"
        st.dataframe(resumo_anual, use_container_width=True)
    
    with col_hist:
        st.subheader("Distribuição das Chuvas > 1 mm")
        valores_chuva = df_filtrado["precipitacao"][df_filtrado["precipitacao"] > 1]
        if not valores_chuva.empty:
            fig_hist = px.histogram(valores_chuva, nbins=40, labels={'value': 'Precipitação (mm)'})
            # ALTERADO: Template do gráfico para 'plotly_dark'
            fig_hist.update_layout(template="plotly_dark", showlegend=False, yaxis_title="Frequência", bargap=0.1, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            fig_hist.update_traces(marker_color='#009E73')
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Nenhuma hora com chuva > 1 mm no período selecionado.")

with tab2:
    st.header("Análise das Máximas Anuais")
    duracao_max = st.selectbox(
        "Selecione a duração para acumulação (horas):", 
        [1, 2, 3, 6, 12, 24], 
        key='duracao_maximas'
    )
    
    with st.spinner(f"Calculando máximas anuais para {duracao_max}h..."):
        maximas_anuais = calculate_annual_maxima(df_filtrado, duracao_max)
    
    if maximas_anuais.empty:
        st.warning("Não foi possível calcular as máximas anuais. Verifique o intervalo de anos selecionado.")
    else:
        df_maximas = maximas_anuais.reset_index()
        df_maximas.columns = ["Ano", f"Precipitação Máxima (mm)"]
        
        fig_maximas = px.bar(
            df_maximas, x="Ano", y="Precipitação Máxima (mm)",
            title=f"Máximas Anuais para Acumulação de {duracao_max}h",
            text_auto=False # ALTERADO: Mudado de '.2s' para False para mostrar valores reais
        )
        # ALTERADO: Template do gráfico para 'plotly_dark'
        fig_maximas.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig_maximas.update_traces(marker_color='#56B4E9')
        st.plotly_chart(fig_maximas, use_container_width=True)
        
        with st.expander("Ver tabela de dados das máximas anuais"):
            st.dataframe(df_maximas.set_index("Ano"), use_container_width=True)

with tab3:
    st.header("Geração de Curvas Intensidade-Duração-Frequência")
    duracao_idf = st.selectbox(
        "Escolha a duração para o ajuste estatístico (horas):", 
        [1, 2, 3, 6, 12, 24], 
        key='duracao_idf'
    )
    
    trs = [2, 5, 10, 25, 50, 100]
    
    with st.spinner(f"Ajustando curvas para duração de {duracao_idf}h..."):
        df_idf, params_gumbel, params_lp3, serie_maximas = calculate_idf_curves(
            calculate_annual_maxima(df_filtrado, duracao_idf), 
            duracao_idf, 
            np.array(trs)
        )

    if df_idf is None:
        st.warning("A série histórica é muito curta para um ajuste estatístico confiável (requer no mínimo 5 anos de dados).")
    else:
        st.session_state['df_idf'] = df_idf
        st.session_state['params_gumbel'] = params_gumbel
        st.session_state['params_lp3'] = params_lp3
        
        fig_idf = go.Figure()
        fig_idf.add_trace(go.Scatter(
            x=df_idf["TR (anos)"], y=df_idf[f"Gumbel_{duracao_idf}h (mm)"], 
            mode='lines+markers', name='Gumbel', line=dict(color='#D55E00')
        ))
        fig_idf.add_trace(go.Scatter(
            x=df_idf["TR (anos)"], y=df_idf[f"LP3_{duracao_idf}h (mm)"],
            mode='lines+markers', name='Log-Pearson III', line=dict(color='#0072B2')
        ))
        # ALTERADO: Template do gráfico para 'plotly_dark'
        fig_idf.update_layout(
            title=f"Precipitação Máxima Estimada vs. Período de Retorno (Duração: {duracao_idf}h)",
            xaxis_title="Período de Retorno (anos)", yaxis_title="Precipitação Máxima (mm)",
            xaxis_type="log", template="plotly_dark",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_idf, use_container_width=True)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            # Salva o gráfico com fundo BRANCO para o PDF
            # Crie uma cópia da figura e altere o template para "plotly_white" ou defina as cores explicitamente
            fig_idf_for_pdf = go.Figure(fig_idf) # Cria uma cópia da figura
            fig_idf_for_pdf.update_layout(template="plotly_white", paper_bgcolor='white', plot_bgcolor='white')
            # Você também pode ajustar a cor do texto e das linhas para serem visíveis em fundo branco
            fig_idf_for_pdf.update_layout(
                font=dict(color="black"),  # Altera a cor do texto para preto
                xaxis=dict(gridcolor="lightgrey", linecolor="black"),
                yaxis=dict(gridcolor="lightgrey", linecolor="black")
            )
            # Garanta que as cores das linhas dos traços também sejam visíveis em fundo branco
            for trace in fig_idf_for_pdf.data:
                if isinstance(trace, go.Scatter):
                    # Se as linhas são muito claras no modo dark, garanta que sejam escuras o suficiente no modo light
                    if trace.line.color == '#D55E00': # Cor original Gumbel
                        trace.line.color = '#D55E00' # Mantém se já for bom, ou muda para algo como 'darkorange'
                    if trace.line.color == '#0072B2': # Cor original LP3
                        trace.line.color = '#0072B2' # Mantém se já for bom, ou muda para algo como 'darkblue'

            fig_idf_for_pdf.write_image(tmpfile.name, scale=2, format="png", engine="kaleido")
            st.session_state['grafico_path'] = tmpfile.name
        
        # --- ALTERADO: AJUSTE ESTATÍSTICO AGORA ACIMA DA TABELA ---
        st.subheader("Ajuste Estatístico")
        
        st.markdown("---")
        st.markdown("### **Distribuição Gumbel**")
        col_mu, col_beta, col_ks = st.columns(3)
        with col_mu:
            st.metric(label="Parâmetro de Posição (mu)", value=f"{params_gumbel['mu']:.2f} mm")
        with col_beta:
            st.metric(label="Parâmetro de Escala (beta)", value=f"{params_gumbel['beta']:.2f} mm")
        with col_ks:
            st.metric(label="Teste K-S (p-valor)", value=f"{params_gumbel['ks_p']:.3f}", 
                      delta="Boa aderência" if params_gumbel['ks_p'] > 0.05 else "Aderência fraca (α=5%)", delta_color="normal")
        
        st.markdown("---")
        st.markdown("### **Distribuição Log-Pearson III**")
        col_mean, col_std, col_skew = st.columns(3)
        with col_mean:
            st.metric(label="Média (log10)", value=f"{params_lp3['mean_log']:.3f}")
        with col_std:
            st.metric(label="Desvio Padrão (log10)", value=f"{params_lp3['std_log']:.3f}")
        with col_skew:
            st.metric(label="Coef. de Assimetria (log10)", value=f"{params_lp3['skew']:.3f}")
        # --- FIM DA ALTERAÇÃO DO AJUSTE ESTATÍSTICO ---

        # Tabela de Resultados (agora abaixo do Ajuste Estatístico)
        st.subheader("Resultados")
        st.dataframe(df_idf.style.format("{:.2f}"))
        
        csv_data = df_idf.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Baixar Tabela (CSV)", csv_data, f"curvas_IDF_{duracao_idf}h.csv", "text/csv")
            
        with st.expander("Ver detalhes da série de máximas anuais"):
            st.write(f"**Série de Máximas Anuais ({duracao_idf}h) utilizada:**")
            st.dataframe(serie_maximas)

with tab4:
    st.header("Gerar Relatório em PDF")
    st.markdown("Após calcular as curvas na aba **'📉 Curvas IDF'**, você pode gerar um relatório consolidado em PDF com os resultados.")

    if 'df_idf' in st.session_state:
        st.success(f"Tudo pronto para gerar o relatório para a duração de **{st.session_state.duracao_idf} horas**.")
        
        if st.button("📄 Gerar e Baixar PDF"):
            with st.spinner("Montando seu relatório..."):
                pdf_obj = generate_professional_pdf(
                    st.session_state['df_idf'],
                    st.session_state['grafico_path'],
                    st.session_state.duracao_idf,
                    st.session_state['params_gumbel'],
                    st.session_state['params_lp3']
                )
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    pdf_obj.output(tmp.name)
                    with open(tmp.name, "rb") as f:
                        st.download_button(
                            "📥 Clique aqui para baixar o Relatório",
                            f.read(),
                            f"Relatorio_IDF_{st.session_state.duracao_idf}h.pdf",
                            "application/pdf"
                        )
                os.remove(st.session_state['grafico_path'])
    else:
        st.warning("Por favor, gere uma curva na aba '📉 Curvas IDF' primeiro para habilitar esta função.")
