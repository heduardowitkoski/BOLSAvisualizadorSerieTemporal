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

    tab1, tab2, tab3_idf, tab4_chuva_projeto, tab5_relatorio = st.tabs([
        "Visão Geral",
        "Máximas Anuais",
        "Curvas IDF",
        "Chuva de Projeto",
        "Relatório PDF"
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

    with tab5_relatorio:
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
