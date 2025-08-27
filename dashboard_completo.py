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
    initial_sidebar_state="collapsed"
)

# --- CSS CUSTOMIZADO COM LINK PARA O FONT AWESOME E ESTILOS PARA ÍCONES ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css');

    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Cores do tema escuro - Mantidas como referência, mas aplicaremos diretamente */
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
        color: var(--text-color); /* Usando a variável para consistência */
    }

    /* Headers */
    h1, h2, h3, [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2 {
        color: #F8F8F8 !important;
        font-weight: 600;
    }
    
    h2 {
        border-bottom: 2px solid var(--border-color);
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    /* Estilo dos cards e containers */
    .stMetric, [data-testid="stMetric"], [data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"], .stDataFrame {
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
        color: #B0B0B0; /* Cinza claro para abas inativas */
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: var(--primary-color);
        border-bottom: 2px solid var(--primary-color);
        font-weight: 600;
    }
    
    /* Alinha o ícone e o texto nas abas */
    .stTabs [data-baseweb="tab"] [data-testid="stMarkdownContainer"] p {
        display: flex;
        align-items: center;
        gap: 8px;
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

    /* INPUTS DE TEXTO E SELECTBOX */
    .stTextInput label, .stSelectbox label, .stNumberInput label, .stRadio label {
        color: var(--text-color) !important;
    }
    .stTextInput div[data-baseweb="input"] input, .stSelectbox div[data-baseweb="select"] div[role="button"], .stNumberInput input {
        color: var(--text-color) !important;
        background-color: var(--secondary-background-color) !important;
        border: 1px solid var(--border-color) !important;
    }

    /* Cor do texto dentro de `st.info` */
    .stAlert p {
        color: var(--text-color) !important;
    }
    
    /* Cor do texto de markdown e `st.write` */
    .stMarkdown p, .stText {
        color: var(--text-color) !important;
    }

    /* Estilo para centralizar métricas */
    .centered-metric [data-testid="stMetricValue"] {
        text-align: center;
        width: 100%;
    }
    .centered-metric [data-testid="stMetricLabel"] {
        text-align: center;
        width: 100%;
    }

    /* ESCONDE A BARRA LATERAL E O BOTÃO DE ABRIR */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="stSidebarToggleButton"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. FUNÇÕES AUXILIARES E CACHE
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
        return None, None, None, series, None, None    
    
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

    return df_idf, params_gumbel, params_lp3, series, (mu_g, beta_g), (mean_log, std_log, skew)

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

# Gerenciamento de estado para o DataFrame
if 'df' not in st.session_state:
    st.session_state['df'] = None

# --- Página Inicial para Carregamento de Arquivo ---
if st.session_state['df'] is None:
    st.title("Sistema de Análise Pluviométrica (SAP-IDF)")
    st.caption("Uma ferramenta para análise de séries históricas de chuva e geração de curvas Intensidade-Duração-Frequência.")
    st.divider()

    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("<h1 style='text-align: center;'><i class='fas fa-cloud-rain fa-3x'></i></h1>", unsafe_allow_html=True)
    with col2:
        st.header("Bem-vindo ao SAP-IDF!")
        st.markdown("""
        Para começar, **carregue seus dados de chuva horária (.csv)**.

        **Passos:**
        1.  Clique em **'Browse files'**.
        2.  Selecione o arquivo.
        3.  Aguarde o processamento.
        """)
    
    uploaded_file = st.file_uploader(
        "Carregue seu arquivo CSV",
        type=["csv"],
        help="O arquivo deve conter colunas 'datahora' e 'precipitacao'."
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Analisando seu arquivo..."):
                st.session_state['df'] = load_data(uploaded_file)
            st.success(f"Arquivo **{uploaded_file.name}** carregado com sucesso!")
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")
            st.session_state['df'] = None

# --- Exibir o dashboard completo apenas se o arquivo foi carregado ---
if st.session_state['df'] is not None:
    df = st.session_state['df']
    
    ano_min_disponivel = df.index.year.min()
    ano_max_disponivel = df.index.year.max()
    df_analise = df 

    tab1, tab2, tab3_idf, tab4_chuva_projeto, tab5_tc, tab6_racional, tab7_circulares, tab8_canais_abertos, tab9_relatorio, = st.tabs([
    "Visão Geral",
    "Máximas Anuais",
    "Curvas IDF",
    "Chuva de Projeto",
    "Tempo de Concentração",
    "Vazão de Projeto (Método Racional)",
    "Dimensionamento de Condutos (Manning)",
    "Canais Abertos (Manning)",
    "Relatório PDF",
    ])

    with tab1:
        st.markdown("## <i class='fas fa-chart-bar'></i> Visão Geral dos Dados", unsafe_allow_html=True)
        
        with st.container():
            col1, col2, col3 = st.columns(3) 
            col1.metric("Período Analisado", f"{ano_min_disponivel}–{ano_max_disponivel}")
            col2.metric("Total de Registros", f"{len(df_analise):,}")
            col3.metric("Máximo Horário", f"{df_analise['precipitacao'].max():.1f} mm")

        st.subheader("Série Temporal da Precipitação (Horária)")
        fig_hourly = go.Figure(data=go.Scatter(x=df_analise.index, y=df_analise['precipitacao'],   
                                                 mode='lines', name='Precipitação Horária',
                                                 line=dict(color='#009E73', width=1)))
        fig_hourly.update_layout(
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',    
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Data",
            yaxis_title="Precipitação (mm)",
            title="Precipitação Horária Registrada"
        )
        st.plotly_chart(fig_hourly, use_container_width=True, config={'displayModeBar': False})
        
        st.subheader("Série Temporal da Precipitação (Agregada)")
        with st.container():
            aggregation_level = st.selectbox(
                "Agregação:", 
                options=['Diária', 'Mensal', 'Anual'],
                index=0
            )

            df_plot = df_analise['precipitacao']
            if aggregation_level == 'Diária':
                df_plot = df_plot.resample('D').sum()
            elif aggregation_level == 'Mensal':
                df_plot = df_plot.resample('ME').sum()
            elif aggregation_level == 'Anual':
                df_plot = df_plot.resample('YE').sum()

            fig_serie = px.line(x=df_plot.index, y=df_plot.values, labels={'y': 'Precipitação (mm)', 'x': 'Data'})
            fig_serie.update_layout(template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            fig_serie.update_traces(line_color= 'var(--primary-color)', line_width=1.5)
            st.plotly_chart(fig_serie, use_container_width=True)

        st.markdown("---")
        st.subheader("Resumos Estatísticos Anuais e Distribuição")
        
        col_resumo, col_hist = st.columns([0.8, 1.2], gap="large") 
        
        with col_resumo:
            st.markdown("**Precipitação Anual Total**")
            resumo_anual = df_analise['precipitacao'].resample("YE").agg(['sum']).rename(
                columns={'sum': 'Total (mm)'}
            )
            resumo_anual.index = resumo_anual.index.year
            resumo_anual.index.name = "Ano"
            st.dataframe(resumo_anual, use_container_width=True, height=250)
        
        with col_hist:
            st.markdown("**Distribuição de Chuvas Horárias**")
            valores_chuva = df_analise["precipitacao"][df_analise["precipitacao"] > 1]
            if not valores_chuva.empty:
                fig_hist = px.histogram(valores_chuva, nbins=40, labels={'value': 'Precipitação (mm)'})
                fig_hist.update_layout(template="plotly_dark", showlegend=False, yaxis_title="Frequência", bargap=0.1, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
                fig_hist.update_traces(marker_color='#009E73')
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("Nenhuma chuva > 1 mm detectada.") 

    with tab2:
        st.markdown("## <i class='fas fa-chart-line'></i> Análise de Máximas Anuais", unsafe_allow_html=True)
        duracao_max = st.selectbox(
            "Duração (horas):",
            [1, 2, 3, 6, 12, 24],
            key='duracao_maximas'
        )
        
        with st.spinner(f"Calculando máximas para {duracao_max}h..."):
            maximas_anuais = calculate_annual_maxima(df_analise, duracao_max)
        
        if maximas_anuais.empty:
            st.warning("Dados insuficientes para máximas anuais.")
        else:
            df_maximas = maximas_anuais.reset_index()
            df_maximas.columns = ["Ano", "Precipitação Máxima (mm)"]
            
            fig_maximas = px.bar(
                df_maximas, x="Ano", y="Precipitação Máxima (mm)",
                title=f"Máximas Anuais ({duracao_max}h)",
                text_auto=".1f"
            )
            fig_maximas.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            fig_maximas.update_traces(marker_color='#56B4E9', textposition='outside')
            st.plotly_chart(fig_maximas, use_container_width=True)
            
            with st.expander("Ver dados"):
                st.dataframe(df_maximas.set_index("Ano"), use_container_width=True)

    with tab3_idf:
        st.markdown("## <i class='fas fa-chart-area'></i> Curvas Intensidade-Duração-Frequência (IDF)", unsafe_allow_html=True)
        st.markdown("Calcule e visualize as **curvas IDF** e use os parâmetros para **chuva de projeto**.")

        duracao_idf = st.selectbox(
            "Duração para ajuste (horas):",
            [1, 2, 3, 6, 12, 24],
            key='duracao_idf'
        )
        
        trs = [2, 5, 10, 25, 50, 100]
        
        with st.spinner(f"Ajustando curvas para {duracao_idf}h..."):
            df_idf, params_gumbel, params_lp3, serie_maximas, gumbel_params_tuple, lp3_params_tuple = calculate_idf_curves(
                calculate_annual_maxima(df_analise, duracao_idf),
                duracao_idf,
                np.array(trs)
            )

        if df_idf is None:
            st.warning("Série curta para ajuste estatístico (mínimo de 5 anos de dados).")
        else:
            st.session_state['gumbel_params'] = gumbel_params_tuple
            st.session_state['lp3_params'] = lp3_params_tuple
            st.session_state['duracao_idf_calculada'] = duracao_idf

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
            fig_idf.update_layout(
                title=f"Precipitação Estimada vs. Período de Retorno (Duração: {duracao_idf}h)",
                xaxis_title="Período de Retorno (anos)", yaxis_title="Precipitação (mm)",
                xaxis_type="log", template="plotly_dark",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_idf, use_container_width=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                fig_idf_for_pdf = go.Figure(fig_idf)    
                fig_idf_for_pdf.update_layout(template="plotly_white", paper_bgcolor='white', plot_bgcolor='white')
                fig_idf_for_pdf.update_layout(
                    font=dict(color="black"),  
                    xaxis=dict(gridcolor="lightgrey", linecolor="black"),
                    yaxis=dict(gridcolor="lightgrey", linecolor="black")
                )
                for trace in fig_idf_for_pdf.data:
                    if isinstance(trace, go.Scatter):
                        if trace.line.color == '#D55E00':    
                            trace.line.color = '#D55E00'    
                        if trace.line.color == '#0072B2':    
                            trace.line.color = '#0072B2'    

                fig_idf_for_pdf.write_image(tmpfile.name, scale=2, format="png")
                st.session_state['grafico_path'] = tmpfile.name
            
            st.subheader("Resultados IDF")
            st.dataframe(df_idf.style.format("{:.2f}"))
            
            csv_data = df_idf.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Baixar Tabela (CSV)", csv_data, f"curvas_IDF_{duracao_idf}h.csv", "text/csv")

            st.markdown("---")
            st.subheader("Ajuste Estatístico")
            
            st.markdown("### **Gumbel**")
            col_mu, col_beta, col_ks = st.columns(3)
            with col_mu:
                st.metric(label="Posição (mu)", value=f"{params_gumbel['mu']:.2f} mm")
            with col_beta:
                st.metric(label="Escala (beta)", value=f"{params_gumbel['beta']:.2f} mm")
            with col_ks:
                st.metric(label="K-S (p-valor)", value=f"{params_gumbel['ks_p']:.3f}",
                                 delta="Boa aderência" if params_gumbel['ks_p'] > 0.05 else "Aderência fraca (α=5%)", delta_color="normal")
            
            st.markdown("---")
            st.markdown("### **Log-Pearson III**")
            col_mean, col_std, col_skew = st.columns(3)
            with col_mean:
                st.metric(label="Média (log10)", value=f"{params_lp3['mean_log']:.3f}")
            with col_std:
                st.metric(label="Desvio Padrão (log10)", value=f"{params_lp3['std_log']:.3f}")
            with col_skew:
                st.metric(label="Assimetria (log10)", value=f"{params_lp3['skew']:.3f}")

    with tab4_chuva_projeto:
        st.markdown("## <i class='fas fa-cloud-showers-heavy'></i> Cálculo de Chuva de Projeto", unsafe_allow_html=True)

        with st.container(border=True):
            st.subheader("Parâmetros de Entrada")
            col_input_params, col_input_method = st.columns([1, 1])

            with col_input_params:
                tr_tab4 = st.number_input(
                    "Período de Retorno (anos):",
                    min_value=1,
                    step=1,
                    value=10,
                    key="tr_tab4_separate"
                )
                dur_tab4 = st.number_input(
                    "Duração da Chuva (horas):",
                    min_value=0.1,
                    step=0.1,
                    value=float(st.session_state.get('duracao_idf_calculada', 1.0)), 
                    key="dur_tab4_separate" 
                )

            with col_input_method:
                metodo_tab4 = st.radio(
                    "Método de Cálculo:",
                    ["Gumbel", "Log-Pearson III"],
                    key="metodo_tab4",
                    horizontal=False
                )
            
            st.markdown("---")
            col_calc_btn_left, col_calc_btn_center, col_calc_btn_right = st.columns([1, 1, 1])
            with col_calc_btn_center:
                if st.button("Calcular Chuva de Projeto", key="btn_calc_tab4_separate", use_container_width=True):
                    chuva_proj = None
                    intensidade_proj = None

                    if metodo_tab4 == "Gumbel":
                        if 'gumbel_params' in st.session_state and st.session_state['gumbel_params'] is not None:
                            mu_value, beta_value = st.session_state['gumbel_params']
                            chuva_proj = gumbel_r.ppf(1 - 1/tr_tab4, loc=mu_value, scale=beta_value)
                        else:
                            st.error("Parâmetros Gumbel não encontrados. Calcule-os na aba 'Curvas IDF'.")
                    else:
                        if 'lp3_params' in st.session_state and st.session_state['lp3_params'] is not None:
                            mean_log_value, std_log_value, skew_value = st.session_state['lp3_params']
                            try:
                                dist_lp3 = pearson3(skew_value, loc=mean_log_value, scale=std_log_value)
                                chuva_proj = 10 ** dist_lp3.ppf(1 - 1/tr_tab4)
                            except Exception as e:
                                st.error(f"Erro no cálculo Log-Pearson III: {e}")
                        else:
                            st.error("Parâmetros Log-Pearson III não encontrados. Calcule-os na aba 'Curvas IDF'.")

                    if chuva_proj is not None:
                        if dur_tab4 > 0:
                            intensidade_proj = chuva_proj / dur_tab4
                            st.session_state['chuva_proj_result'] = chuva_proj
                            st.session_state['intensidade_proj_result'] = intensidade_proj
                            st.session_state['show_project_results'] = True
                        else:
                            st.error("A Duração da Chuva deve ser maior que zero.")
                            st.session_state['show_project_results'] = False
                    else:
                        st.warning("Não foi possível calcular a chuva de projeto. Verifique os parâmetros na aba 'Curvas IDF'.")
                        st.session_state['show_project_results'] = False

        if st.session_state.get('show_project_results', False):
            st.markdown("---")
            with st.container(border=True):
                st.subheader("Resultados do Cálculo")
                col_res_chuva, col_res_int = st.columns(2)
                
                with col_res_chuva:
                    st.metric(label="Chuva de Projeto (mm)", value=f"{st.session_state['chuva_proj_result']:.2f}")
                with col_res_int:
                    st.metric(label="Intensidade (mm/h)", value=f"{st.session_state['intensidade_proj_result']:.2f}")
        else:
            st.markdown("---")
            st.info("Aguardando o cálculo da chuva de projeto.")

    with tab9_relatorio:
        st.markdown("## <i class='fas fa-file-alt'></i> Relatório em PDF", unsafe_allow_html=True)
        st.markdown("Gere um relatório PDF após calcular as curvas na aba **'Curvas IDF'**.")

        if 'df_idf' in st.session_state and st.session_state['df_idf'] is not None and 'grafico_path' in st.session_state:
            st.success(f"Pronto para gerar relatório para a duração de **{st.session_state.duracao_idf} horas**.")
            
            if st.button("Gerar Relatório PDF"):
                with st.spinner("Gerando relatório..."):
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
                                "Baixar Relatório",
                                f.read(),
                                f"Relatorio_IDF_{st.session_state.duracao_idf}h.pdf",
                                "application/pdf"
                            )
                    os.remove(st.session_state['grafico_path'])
        else:
            st.warning("Por favor, calcule uma curva IDF na aba 'Curvas IDF' primeiro para gerar o relatório.")




    with tab5_tc:
        st.markdown("## 🕒 Tempo de Concentração")
        st.caption("Cálculo do tempo de concentração utilizando fórmulas empíricas (Kirpich e Giandotti).")

        metodo_tc = st.selectbox("Selecione o método de cálculo:", ["Kirpich", "Giandotti"])

        if metodo_tc == "Kirpich":
            st.markdown("### Método de Kirpich")
            with st.container(border=True):
                c1, c2 = st.columns(2)
                with c1:
                    L_kirpich = st.number_input("Comprimento do curso d'água (m)", min_value=1.0, value=500.0, step=10.0)
                with c2:
                    i_kirpich = st.number_input("Declividade média (m/m)", min_value=0.001, value=0.02, step=0.001, format="%.3f")

            if i_kirpich <= 0:
                st.warning("A declividade deve ser maior que zero.")
            else:
                # Tc(min) = 0.0195 * L^0.77 * i^-0.385  (L em m; i em m/m)
                Tc_kirpich_min = 0.0195 * (L_kirpich ** 0.77) * (i_kirpich ** -0.385)
                st.success(f"Tempo de concentração (Kirpich): **{Tc_kirpich_min:.2f} minutos**")

                # Armazena em estado para uso posterior (relatório/continuidade do cálculo)
            st.session_state["tc_min"] = Tc_kirpich_min
            st.session_state["tc_parametros"] = {
                "método": "Kirpich",
                "L (m)": f"{L_kirpich:.2f}",
                "declividade i (m/m)": f"{i_kirpich:.4f}",
                "Tc (min)": f"{Tc_kirpich_min:.2f}",
                "Tc (h)": f"{Tc_kirpich_min/60:.2f}",
            }

        elif metodo_tc == "Giandotti":
            st.markdown("### Método de Giandotti")
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    A_giandotti = st.number_input("Área da bacia (km²)", min_value=0.01, value=0.50, step=0.01)
                with c2:
                    L_giandotti = st.number_input("Comprimento do percurso principal (km)", min_value=0.10, value=1.00, step=0.10)
                with c3:
                    Hmax = st.number_input("Cota máxima da bacia (m)", value=120.0)
                with c4:
                    Hmin = st.number_input("Cota mínima da bacia (m)", value=100.0)

            deltaH = Hmax - Hmin
            if deltaH <= 0:
                st.warning("A cota máxima deve ser maior que a mínima.")
            else:
                # Mantido conforme seu código original:
                # Tc(h) = ((4*A) + (1.5*L)) / (0.8 * ΔH)
                Tc_giandotti_h = ((4 * A_giandotti) + (1.5 * L_giandotti)) / (0.8 * deltaH)
                Tc_giandotti_min = Tc_giandotti_h * 60.0  # minutos
                st.success(f"Tempo de concentração (Giandotti): **{Tc_giandotti_min:.2f} minutos**")

                # Armazena em estado para uso posterior
                st.session_state["tc_min"] = Tc_giandotti_min
                st.session_state["tc_parametros"] = {
                    "método": "Giandotti",
                    "Área A (km²)": f"{A_giandotti:.3f}",
                    "Comprimento L (km)": f"{L_giandotti:.3f}",
                    "ΔH (m)": f"{deltaH:.2f}",
                    "Tc (min)": f"{Tc_giandotti_min:.2f}",
                    "Tc (h)": f"{Tc_giandotti_h:.2f}",
                }



    with tab6_racional:
        st.markdown("## 💧 Vazão de Projeto – Método Racional")
        st.caption("Cálculo da vazão de projeto usando a fórmula: `Q = C × i × A`")

        # Recuperar chuva de projeto (intensidade) e tempo de concentração se disponíveis
        intensidade_chuva = st.session_state.get("intensidade_proj_result", None)  # em mm/h
        tempo_concentracao = st.session_state.get("tc_min", None)              # em minutos

        # Mostrar informações recuperadas
        if tempo_concentracao:
            st.info(f"Tempo de concentração recuperado: **{tempo_concentracao:.2f} minutos**")
        else:
            st.warning("Tempo de concentração não disponível. Calcule na aba anterior.")

        if intensidade_chuva:
            st.info(f"Chuva de projeto recuperada: **{intensidade_chuva:.2f} mm/h**")
        else:
            st.warning("Chuva de projeto não disponível. Calcule na aba correspondente.")

        # Entradas do usuário
        C = st.number_input("Coeficiente de escoamento (C)", min_value=0.1, max_value=1.0, value=0.6, step=0.05)
        A = st.number_input("Área da bacia de contribuição (ha)", min_value=0.01, value=5.0, step=0.1)

        # Cálculo
        if intensidade_chuva and tempo_concentracao:
            Q = C * intensidade_chuva * A / 360.0  # (C adimensional) × (i mm/h) × (A ha) / 360 → m³/s
            st.success(f"**Vazão de projeto estimada: {Q:.3f} m³/s**")

            # salva para usar em outras abas
            st.session_state["q_projeto"] = Q  

            # guarda parâmetros para o relatório
            st.session_state["racional_parametros"] = {
                "C (adim)": f"{C:.2f}",
                "Área A (ha)": f"{A:.2f}",
                "Intensidade i (mm/h)": f"{intensidade_chuva:.2f}",
                "Tc (min)": f"{tempo_concentracao:.2f}",
                "Q (m³/s)": f"{Q:.3f}",
            }
        else:
            st.info("Aguardando definição da chuva de projeto e tempo de concentração.")



    with tab7_circulares:
        import math

        st.markdown("## 📏 Dimensionamento de Condutos – Fórmula de Manning")
        st.caption("Estimativa do diâmetro necessário para escoar a vazão de projeto utilizando seção **circular cheia**.")

        # ---------------------------
        # Recuperar vazão de projeto
        # ---------------------------
        Q = st.session_state.get("q_projeto", None)
        if Q and Q > 0:
            st.info(f"Vazão de projeto recuperada: **{Q:.3f} m³/s**")
        else:
            st.warning("Vazão de projeto não encontrada. Calcule-a na aba **Vazão de Projeto (Método Racional)**.")
            # permite informar manualmente se desejar
            Q = st.number_input("Informe manualmente Q (m³/s)", min_value=0.0, value=0.0, step=0.001, format="%.6f")

        # ---------------------------
        # Entradas do usuário
        # ---------------------------
        with st.container(border=True):
            st.subheader("Parâmetros")
            c1, c2, c3 = st.columns(3)
            with c1:
                n = st.number_input("Coeficiente de Manning (n)", min_value=0.010, max_value=0.030, value=0.013, step=0.001, format="%.3f")
            with c2:
                S = st.number_input("Declividade do conduto S (m/m)", min_value=0.0001, value=0.0100, step=0.0001, format="%.4f")
            with c3:
                passo_cm = st.number_input("Passo de busca (cm)", min_value=0.1, value=1.0, step=0.1, help="Resolução do varrimento de diâmetros.")
            c4, c5 = st.columns(2)
            with c4:
                d_min = st.number_input("Diâmetro mínimo (m)", min_value=0.05, value=0.10, step=0.01)
            with c5:
                d_max = st.number_input("Diâmetro máximo (m)", min_value=0.10, value=3.00, step=0.05)

        # ---------------------------
        # Núcleo de cálculo
        # ---------------------------
        def q_manning_circular_cheia(d, n, S):
            """Q = (1/n) * A * R^(2/3) * S^(1/2), com R = D/4 e A = π D²/4 (seção cheia)."""
            if d <= 0 or n <= 0 or S <= 0:
                return 0.0
            R = d / 4.0
            A = (math.pi / 4.0) * d * d
            return (1.0/n) * A * (R ** (2.0/3.0)) * (S ** 0.5)

        # ---------------------------
        # Varrimento do diâmetro
        # ---------------------------
        resultado = None
        if Q and Q > 0:
            passo_m = passo_cm / 100.0
            # garante limites coerentes
            d_min = max(0.01, float(d_min))
            d_max = max(d_min + passo_m, float(d_max))

            d = d_min
            while d <= d_max + 1e-9:
                q_est = q_manning_circular_cheia(d, n, S)
                if q_est >= Q:
                    resultado = (d, q_est)
                    break
                d += passo_m

        # ---------------------------
        # Saída + métricas + session_state para relatório
        # ---------------------------
        G = 9.81
        RHO = 1000.0

        if resultado:
            d_rec, Q_calc = resultado
            A = (math.pi / 4.0) * d_rec * d_rec
            R = d_rec / 4.0
            V = Q_calc / A if A > 0 else float("nan")
            tau = RHO * G * R * S  # Pa

            st.success(f"**Diâmetro mínimo recomendado: {d_rec:.3f} m**")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Q calculada (m³/s)", f"{Q_calc:.6f}")
                st.metric("Área A (m²)", f"{A:.4f}")
            with c2:
                st.metric("Velocidade V (m/s)", f"{V:.3f}")
                st.metric("Raio hidráulico R (m)", f"{R:.4f}")
            with c3:
                st.metric("Declividade S (m/m)", f"{S:.4f}")
                st.metric("Tensão média τ (Pa)", f"{tau:.1f}")

            # salva resumo para o Relatório (tab9)
            st.session_state["resultado_circulares"] = {
                "diâmetro D (m)": f"{d_rec:.3f}",
                "n (Manning)": f"{n:.3f}",
                "declividade S (m/m)": f"{S:.5f}",
                "Q de projeto (m³/s)": f"{Q:.3f}",
                "Q calculada (m³/s)": f"{Q_calc:.4f}",
                "velocidade V (m/s)": f"{V:.3f}",
                "raio hidráulico R (m)": f"{R:.4f}",
                "tensão média τ (Pa)": f"{tau:.1f}",
            }

        else:
            if Q and Q > 0:
                st.error("Nenhum diâmetro no intervalo informado atingiu a vazão de projeto.")
            else:
                st.info("Informe **Q** (m³/s) na aba de Vazão de Projeto, ou defina manualmente acima para prosseguir.")




    with tab8_canais_abertos:
        import math
        st.markdown("## 🌊 Canais Abertos (Manning)")
        st.caption("Verificação e dimensionamento de seções retangular, trapezoidal e triangular para vazão de projeto.")

        # ---------------------------
        # Estado / vazão de projeto
        # ---------------------------
        Q = st.session_state.get("q_projeto", None)
        if Q and Q > 0:
            st.info(f"Vazão de projeto recuperada: **{Q:.3f} m³/s**")
        else:
            st.warning("Vazão de projeto não encontrada. Calcule-a na aba **Vazão de Projeto (Método Racional)**.")
            Q = st.number_input("Informe manualmente uma vazão de projeto Q (m³/s)", min_value=0.0, value=0.0, step=0.001, format="%.6f")

        # ---------------------------
        # Parâmetros hidráulicos
        # ---------------------------
        with st.container(border=True):
            st.subheader("Parâmetros Hidráulicos")
            c1, c2 = st.columns(2)
            with c1:
                S = st.number_input("Declividade do fundo (m/m)", min_value=0.000001, value=0.001, step=0.0001, format="%.6f")
            with c2:
                materiais = {
                    "Canal de concreto acabado": 0.013,
                    "Concreto rústico": 0.017,
                    "PVC liso (calha aberta)": 0.009,
                    "Terra alisada": 0.022,
                    "Terra média": 0.025,
                    "Terra irregular": 0.030,
                    "Rochoso/encosto natural": 0.035,
                    "Vegetação leve": 0.040,
                    "Vegetação densa": 0.070,
                    "Outro (personalizado)": 0.015,
                }
                mat = st.selectbox("Material (n típico)", list(materiais.keys()), index=0)
                n = st.number_input("Manning n", min_value=0.005, value=float(materiais[mat]), step=0.001, format="%.3f")

        # ---------------------------
        # Núcleo hidráulico
        # ---------------------------
        G = 9.81      # m/s²
        RHO = 1000.0  # kg/m³

        def geom_trapezio(b, z, y):
            """A (m²), P (m), T (m) — trapezoidal (gera ret e tri como casos)."""
            if y <= 0:
                return 0.0, 0.0, max(b, 0.0)
            A = y * (b + z * y)                 # A = b*y + z*y^2
            P = b + 2.0 * y * (1.0 + z**2) ** 0.5
            T = b + 2.0 * z * y
            return A, P, T

        def manning_Q(A, P, S, n):
            if A <= 0 or P <= 0 or S <= 0 or n <= 0:
                return 0.0
            R = A / P
            return (1.0/n) * A * (R ** (2.0/3.0)) * (S ** 0.5)

        def froude(Qx, A, T):
            if A <= 0 or T <= 0:
                return float('nan')
            V = Qx / A
            D = A / T
            denom = (G * D) ** 0.5
            return V / denom if denom > 0 else float('inf')

        def tau_medio(R, S):
            return RHO * G * R * S  # Pa

        def bissecao(f, a, b, tol=1e-6, maxit=100):
            fa, fb = f(a), f(b)
            if math.isnan(fa) or math.isnan(fb) or fa * fb > 0:
                return None
            L, Rr = a, b
            for _ in range(maxit):
                m = 0.5 * (L + Rr)
                fm = f(m)
                if abs(fm) < tol or (Rr - L) < tol:
                    return max(m, 0.0)
                if fa * fm < 0:
                    Rr, fb = m, fm
                else:
                    L, fa = m, fm
            return max(0.5 * (L + Rr), 0.0)

        def y_normal(Qd, b, z, S, n, y_min=1e-4, y_max=50.0):
            def f(y):
                A, P, _ = geom_trapezio(b, z, y)
                return manning_Q(A, P, S, n) - Qd
            fmin, fmax, grow = f(y_min), f(y_max), 0
            while (fmin * fmax > 0) and (grow < 10):
                y_max *= 2.0; fmax = f(y_max); grow += 1
            return bissecao(f, y_min, y_max)

        def y_critico(Qd, b, z, y_min=1e-4, y_max=50.0):
            def F(y):
                A, _, T = geom_trapezio(b, z, y)
                return froude(Qd, A, T) - 1.0
            Fmin, Fmax, grow = F(y_min), F(y_max), 0
            while (math.isnan(Fmin) or math.isnan(Fmax) or Fmin * Fmax > 0) and (grow < 10):
                y_max *= 2.0; Fmax = F(y_max); grow += 1
            return bissecao(F, y_min, y_max)

        def b_para_Q(Qd, z, y, S, n, b_min=0.01, b_max=50.0):
            def f(b):
                A, P, _ = geom_trapezio(b, z, y)
                return manning_Q(A, P, S, n) - Qd
            fmin, fmax, grow = f(b_min), f(b_max), 0
            while (fmin * fmax > 0) and (grow < 10):
                b_max *= 2.0; fmax = f(b_max); grow += 1
            return bissecao(f, b_min, b_max)

        # ---------------------------
        # Geometria + Modo de cálculo
        # ---------------------------
        with st.container(border=True):
            st.subheader("Geometria da Seção")
            tipo = st.selectbox("Tipo de seção", ["Retangular", "Trapezoidal", "Triangular"])

            if tipo == "Retangular":
                z = 0.0
                c1, c2 = st.columns(2)
                with c1:
                    b = st.number_input("Largura da base b (m)", min_value=0.0, value=1.00, step=0.05)
                with c2:
                    y_inf = st.number_input("Profundidade y (m) (para verificação)", min_value=0.0, value=0.80, step=0.05)

            elif tipo == "Triangular":
                b = 0.0
                c1, c2 = st.columns(2)
                with c1:
                    z = st.number_input("Talude lateral z (H:V)", min_value=0.0, value=1.50, step=0.25, help="Ex.: z=1.5 → 1,5 horizontal para 1 vertical (cada lado).")
                with c2:
                    y_inf = st.number_input("Profundidade y (m) (para verificação)", min_value=0.0, value=0.80, step=0.05)

            else:  # Trapezoidal
                c1, c2, c3 = st.columns(3)
                with c1:
                    b = st.number_input("Largura da base b (m)", min_value=0.0, value=1.00, step=0.05)
                with c2:
                    z = st.number_input("Talude lateral z (H:V)", min_value=0.0, value=1.50, step=0.25)
                with c3:
                    y_inf = st.number_input("Profundidade y (m) (para verificação)", min_value=0.0, value=0.80, step=0.05)

        with st.container(border=True):
            st.subheader("Modo de Cálculo")
            modo = st.radio(
                "Selecione a operação",
                ["Verificar seção (Q_calc, V, Fr, τ...)", "Dimensionar profundidade (y) para Q", "Dimensionar largura (b) para Q"],
                horizontal=True
            )

        # ---------------------------
        # Resultados
        # ---------------------------
        with st.container(border=True):
            st.subheader("Resultados")

            if modo == "Verificar seção (Q_calc, V, Fr, τ...)":
                A, P, T = geom_trapezio(b, z, y_inf)
                Q_calc = manning_Q(A, P, S, n)
                V = (Q_calc / A) if A > 0 else float('nan')
                R = (A / P) if P > 0 else float('nan')
                Fr = froude(Q_calc, A, T)
                tau = tau_medio(R, S)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Q calculada (m³/s)", f"{Q_calc:.6f}")
                    st.metric("Área molhada A (m²)", f"{A:.4f}")
                with c2:
                    st.metric("Velocidade V (m/s)", f"{V:.4f}")
                    st.metric("Raio hidráulico R (m)", f"{R:.4f}")
                with c3:
                    st.metric("Froude Fr (-)", f"{Fr:.4f}")
                    st.metric("Tensão média τ (Pa)", f"{tau:.1f}")

                regime = "subcrítico (Fr < 1)" if Fr < 1.0 else ("crítico (≈1)" if abs(Fr-1.0) <= 0.05 else "supercrítico (Fr > 1)")
                st.info(f"**Regime estimado:** {regime}")

                y_norm = y_normal(Q if Q > 0 else Q_calc, b, z, S, n)
                y_crit = y_critico(Q if Q > 0 else Q_calc, b, z)

                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Profundidade normal** (para Q={'Q' if Q>0 else 'Q_calc'}): {('%.4f m' % y_norm) if y_norm else 'não encontrada'}")
                with c2:
                    st.write(f"**Profundidade crítica** (para Q={'Q' if Q>0 else 'Q_calc'}): {('%.4f m' % y_crit) if y_crit else 'não encontrada'}")

                if Q > 0:
                    ok = Q_calc >= Q
                    st.warning(
                        f"A seção **{'ATENDE' if ok else 'NÃO atende'}** a vazão solicitada Q={Q:.6f} m³/s "
                        f"(capacidade calculada {Q_calc:.6f} m³/s)."
                    )

                # >>> salva para Relatório
                st.session_state["resultado_canais_abertos"] = {
                    "tipo_seção": tipo,
                    "b (m)": f"{b:.3f}",
                    "z (H:V)": f"{z:.2f}",
                    "y (m)": f"{y_inf:.3f}",
                    "n (Manning)": f"{n:.3f}",
                    "declividade S (m/m)": f"{S:.5f}",
                    "Q de projeto (m³/s)": f"{(Q if Q>0 else Q_calc):.3f}",
                    "Q calculada (m³/s)": f"{Q_calc:.4f}",
                    "velocidade V (m/s)": f"{V:.3f}",
                    "raio hidráulico R (m)": f"{R:.4f}",
                    "Froude Fr": f"{Fr:.3f}",
                    "tensão média τ (Pa)": f"{tau:.1f}",
                    "profundidade normal (m)": (f"{y_norm:.4f}" if y_norm else "n/d"),
                    "profundidade crítica (m)": (f"{y_crit:.4f}" if y_crit else "n/d"),
                }

            elif modo == "Dimensionar profundidade (y) para Q":
                if Q <= 0:
                    st.error("Defina uma vazão de projeto Q > 0 (na aba de Vazão de Projeto) ou informe manualmente acima.")
                else:
                    yn = y_normal(Q, b, z, S, n)
                    if yn is None:
                        st.error("Não foi possível encontrar a profundidade normal. Verifique S, n, b, z e Q.")
                    else:
                        A, P, T = geom_trapezio(b, z, yn)
                        V = Q / A
                        R = A / P
                        Fr = froude(Q, A, T)
                        tau = tau_medio(R, S)

                        st.success(f"**Profundidade normal encontrada:** y = {yn:.4f} m")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Área A (m²)", f"{A:.4f}")
                            st.metric("Largura de superfície T (m)", f"{T:.4f}")
                        with c2:
                            st.metric("Velocidade V (m/s)", f"{V:.4f}")
                            st.metric("Raio hidráulico R (m)", f"{R:.4f}")
                        with c3:
                            st.metric("Froude Fr (-)", f"{Fr:.4f}")
                            st.metric("Tensão média τ (Pa)", f"{tau:.1f}")

                        yc = y_critico(Q, b, z)
                        st.write(f"**Profundidade crítica** (Fr≈1): {('%.4f m' % yc) if yc else 'não encontrada'}")

                        # >>> salva para Relatório
                        st.session_state["resultado_canais_abertos"] = {
                            "tipo_seção": tipo,
                            "b (m)": f"{b:.3f}",
                            "z (H:V)": f"{z:.2f}",
                            "y (m)": f"{yn:.3f}",
                            "n (Manning)": f"{n:.3f}",
                            "declividade S (m/m)": f"{S:.5f}",
                            "Q de projeto (m³/s)": f"{Q:.3f}",
                            "Q calculada (m³/s)": f"{manning_Q(A, P, S, n):.4f}",
                            "velocidade V (m/s)": f"{V:.3f}",
                            "raio hidráulico R (m)": f"{R:.4f}",
                            "Froude Fr": f"{Fr:.4f}",
                            "tensão média τ (Pa)": f"{tau:.1f}",
                            "profundidade normal (m)": f"{yn:.4f}",
                            "profundidade crítica (m)": (f"{yc:.4f}" if yc else "n/d"),
                        }

            else:  # "Dimensionar largura (b) para Q"
                if Q <= 0:
                    st.error("Defina uma vazão de projeto Q > 0 (na aba de Vazão de Projeto) ou informe manualmente acima.")
                else:
                    y_proj = st.number_input("Profundidade de projeto y (m)", min_value=0.0, value=max(y_inf, 0.50), step=0.05)
                    if tipo == "Triangular":
                        st.info("Para seção triangular (b=0), dimensione **y** em vez de **b**.")
                    else:
                        b_sol = b_para_Q(Q, z, y_proj, S, n)
                        if b_sol is None:
                        	st.error("Não foi possível encontrar a largura b. Verifique S, n, y e Q.")
                        else:
                            A, P, T = geom_trapezio(b_sol, z, y_proj)
                            V = Q / A
                            R = A / P
                            Fr = froude(Q, A, T)
                            tau = tau_medio(R, S)

                            st.success(f"**Largura de base encontrada:** b = {b_sol:.4f} m")
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.metric("Área A (m²)", f"{A:.4f}")
                                st.metric("Largura de superfície T (m)", f"{T:.4f}")
                            with c2:
                                st.metric("Velocidade V (m/s)", f"{V:.4f}")
                                st.metric("Raio hidráulico R (m)", f"{R:.4f}")
                            with c3:
                                st.metric("Froude Fr (-)", f"{Fr:.4f}")
                                st.metric("Tensão média τ (Pa)", f"{tau:.1f}")

                            yc = y_critico(Q, b_sol, z)
                            st.write(f"**Profundidade crítica** (Fr≈1): {('%.4f m' % yc) if yc else 'não encontrada'}")

                            # >>> salva para Relatório
                            st.session_state["resultado_canais_abertos"] = {
                                "tipo_seção": tipo,
                                "b (m)": f"{b_sol:.3f}",
                                "z (H:V)": f"{z:.2f}",
                                "y (m)": f"{y_proj:.3f}",
                                "n (Manning)": f"{n:.3f}",
                                "declividade S (m/m)": f"{S:.5f}",
                                "Q de projeto (m³/s)": f"{Q:.3f}",
                                "Q calculada (m³/s)": f"{manning_Q(A, P, S, n):.4f}",
                                "velocidade V (m/s)": f"{V:.3f}",
                                "raio hidráulico R (m)": f"{R:.4f}",
                                "Froude Fr": f"{Fr:.4f}",
                                "tensão média τ (Pa)": f"{tau:.1f}",
                                "profundidade normal (m)": (f"{y_normal(Q, b_sol, z, S, n):.4f}" if Q>0 else "n/d"),
                                "profundidade crítica (m)": (f"{yc:.4f}" if yc else "n/d"),
                            }

        st.caption("Obs.: válidas para escoamento uniforme. Verifique limites admissíveis de velocidade e tensão conforme material/solo do canal.")








