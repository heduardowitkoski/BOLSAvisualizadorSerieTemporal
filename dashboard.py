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
# 1. CONFIGURAÇÃO DA PÁGINA E ESTILO
# ==============================================================================
st.set_page_config(
    page_title="Análise Pluviométrica (SAP-IDF)",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS para pequenos ajustes visuais (opcional)
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
    }
</style>""", unsafe_allow_html=True)


# ==============================================================================
# 2. FUNÇÕES AUXILIARES E CACHE
# ==============================================================================

# --- Funções de Processamento de Dados (com cache para performance) ---

@st.cache_data
def load_data(uploaded_file):
    """Lê e processa o arquivo CSV de chuvas."""
    df_raw = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8')
    df_raw.columns = [col.strip().lower() for col in df_raw.columns]

    # Normaliza a coluna de precipitação
    if "precipitacao" in df_raw.columns:
        df_raw["precipitacao"] = pd.to_numeric(
            df_raw["precipitacao"].astype(str).str.replace(",", "."), errors='coerce'
        )
    else:
        raise ValueError("Coluna 'precipitacao' não encontrada.")

    # Constrói o datetime
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
    if len(series) < 5:  # Mínimo para um ajuste razoável
        return None, None, None, None
    
    # Gumbel
    mu_g, beta_g = gumbel_r.fit(series.values)
    ks_stat, ks_p = kstest(series.values, 'gumbel_r', args=(mu_g, beta_g))
    ad_result = anderson((series.values - mu_g) / beta_g, dist='gumbel')

    # Log-Pearson III
    dados_log = np.log10(series.values[series.values > 0]) # Evita log(0)
    skew = pd.Series(dados_log).skew()
    mean_log = np.mean(dados_log)
    std_log = np.std(dados_log, ddof=1)
    
    intensities_gumbel = []
    intensities_lp3 = []

    for tr in trs_np:
        F = 1 - 1 / tr
        
        # Gumbel
        x_g = gumbel_r.ppf(F, loc=mu_g, scale=beta_g)
        intensities_gumbel.append(x_g)
        
        # Log-Pearson III
        lp3_dist = pearson3(skew, loc=mean_log, scale=std_log)
        x_lp3_log = lp3_dist.ppf(F)
        intensities_lp3.append(10 ** x_lp3_log)

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

# --- Classe para Geração de PDF ---

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
    
    # Título
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"Curvas IDF para Duração de {duration} horas", ln=True, align='C')
    pdf.ln(10)

    # Gráfico
    pdf.image(fig_path, x=10, y=None, w=190)
    pdf.ln(5)

    # Tabela de Resultados
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Tabela de Precipitações e Intensidades", ln=True)
    pdf.set_font("Arial", '', 10)
    
    # Cabeçalho da tabela
    pdf.set_fill_color(220, 220, 220)
    col_widths = [25, 40, 40, 45, 40]
    headers = ["TR (anos)", "Gumbel (mm)", "LP3 (mm)", "Int. Gumbel (mm/h)", "Int. LP3 (mm/h)"]
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, 1, 0, 'C', 1)
    pdf.ln()

    # Corpo da tabela
    for _, row in df_idf.iterrows():
        pdf.cell(col_widths[0], 8, f"{row.iloc[0]}", 1, 0, 'C')
        pdf.cell(col_widths[1], 8, f"{row.iloc[1]:.2f}", 1, 0, 'C')
        pdf.cell(col_widths[2], 8, f"{row.iloc[2]:.2f}", 1, 0, 'C')
        pdf.cell(col_widths[3], 8, f"{row.iloc[3]:.2f}", 1, 0, 'C')
        pdf.cell(col_widths[4], 8, f"{row.iloc[4]:.2f}", 1, 1, 'C')
    pdf.ln(10)

    # Parâmetros e Testes
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Parâmetros Estatísticos e Testes de Aderência", ln=True)
    pdf.set_font("Arial", '', 10)
    
    # ***** LINHA CORRIGIDA ABAIXO *****
    pdf.multi_cell(0, 6, 
        f"Distribuição Gumbel:\n"
        f"  - Parâmetro de Posição (mu): {params_gumbel['mu']:.2f} mm\n"  # Alterado de 'μ' para 'mu'
        f"  - Parâmetro de Escala (beta): {params_gumbel['beta']:.2f} mm\n" # Alterado de 'β' para 'beta' por consistência
        f"  - Teste K-S (p-valor): {params_gumbel['ks_p']:.4f} "
        f"({'Aceito' if params_gumbel['ks_p'] > 0.05 else 'Rejeitado'} a 5% de significância)\n"
    )
    # ***** FIM DA CORREÇÃO *****

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

# --- Título e Descrição ---
st.title("💧 Sistema de Análise Pluviométrica (SAP-IDF)")
st.markdown(
    "**Bem-vindo!** Faça o upload dos seus dados de chuva horária para visualizar séries temporais, "
    "analisar máximas anuais e gerar curvas Intensidade-Duração-Frequência (IDF) com testes estatísticos."
)
st.divider()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configurações")
    uploaded_file = st.file_uploader(
        "Carregue seu arquivo CSV",
        type=["csv"],
        help="O arquivo deve conter as colunas 'datahora' e 'precipitacao', ou 'data', 'hora' e 'precipitacao'."
    )

# --- Corpo Principal ---
if uploaded_file is None:
    st.info("👈 Por favor, carregue um arquivo CSV para iniciar a análise.")
    st.stop()

try:
    with st.spinner("Processando arquivo... Isso pode levar alguns segundos."):
        df = load_data(uploaded_file)
    st.success(f"✅ Arquivo **{uploaded_file.name}** carregado com sucesso!")
except Exception as e:
    st.error(f"❌ Erro ao processar o arquivo: {e}")
    st.stop()


# --- Filtros na Sidebar (após carregar o dado) ---
with st.sidebar:
    st.divider()
    anos_disponiveis = sorted(df.index.year.unique())
    ano_min_selecionado = st.selectbox("Ano inicial", anos_disponiveis, index=0)
    ano_max_selecionado = st.selectbox("Ano final", anos_disponiveis, index=len(anos_disponiveis)-1)
    
    if ano_min_selecionado > ano_max_selecionado:
        st.warning("O ano inicial deve ser menor ou igual ao ano final.")
        st.stop()

# Filtra o DataFrame principal com base na seleção
df_filtrado = df[(df.index.year >= ano_min_selecionado) & (df.index.year <= ano_max_selecionado)]


# --- Abas de Análise ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Resumo Geral e Série Temporal", 
    "📈 Máximas Anuais", 
    "📉 Curvas IDF", 
    "📄 Relatório PDF"
])

# ==============================================================================
# Aba 1: Resumo Geral (CÓDIGO CORRIGIDO E OTIMIZADO)
# ==============================================================================
with tab1:
    st.subheader("Visão Geral do Período Selecionado")
    
    # Métricas principais (isso é rápido, pode manter)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Período Analisado", f"{ano_min_selecionado}–{ano_max_selecionado}")
    col2.metric("Total de Registros", f"{len(df_filtrado):,}")
    col3.metric("Precipitação Total", f"{df_filtrado['precipitacao'].sum():,.1f} mm")
    col4.metric("Máximo Horário", f"{df_filtrado['precipitacao'].max():.1f} mm")
    st.divider()

    # --- GRÁFICO OTIMIZADO ---
    st.subheader("Série Temporal da Precipitação")

    # Adiciona um seletor de agregação
    aggregation_level = st.selectbox(
        "Selecione a resolução do gráfico:",
        options=['Diária', 'Semanal', 'Mensal', 'Horária (somente para períodos curtos)'],
        index=0  # Padrão para 'Diária'
    )

    # Lógica de agregação
    df_plot = df_filtrado['precipitacao']
    if aggregation_level == 'Diária':
        df_plot = df_plot.resample('D').sum()
        title = "Precipitação Diária Acumulada"
    elif aggregation_level == 'Semanal':
        df_plot = df_plot.resample('W').sum()
        title = "Precipitação Semanal Acumulada"
    elif aggregation_level == 'Mensal':
        df_plot = df_plot.resample('ME').sum()
        title = "Precipitação Mensal Acumulada"
    else: # Horária
        # AVISO: Só permita a visualização horária se o número de pontos for razoável
        if len(df_filtrado) > 5000: # Ex: ~7 meses de dados horários
            st.warning("A visualização horária foi desativada pois o período selecionado é muito longo. Por favor, escolha outra resolução ou selecione um intervalo de anos menor.")
            st.stop()
        title = "Precipitação Horária"

    # Plota o gráfico com os dados agregados (ou não)
    fig_serie = px.line(
        x=df_plot.index, y=df_plot.values,
        title=title,
        labels={'y': 'Precipitação (mm)', 'x': 'Data'},
        template="plotly_white"
    )
    fig_serie.update_traces(line_color='#0072B2', line_width=1)
    st.plotly_chart(fig_serie, use_container_width=True)


    # --- Restante da Aba 1 ---
    st.divider()
    col_resumo, col_hist = st.columns(2)
    with col_resumo:
        st.subheader("Resumo Anual")
        # CORREÇÃO DO FUTURE WARNING: 'Y' -> 'YE'
        resumo_anual = df_filtrado['precipitacao'].resample("YE").agg(['sum', 'max']).rename(
            columns={'sum': 'Total (mm)', 'max': 'Máximo Horário (mm)'}
        )
        resumo_anual.index = resumo_anual.index.year
        st.dataframe(resumo_anual, use_container_width=True)
    
    with col_hist:
        st.subheader("Distribuição das Chuvas > 1 mm")
        valores_chuva = df_filtrado["precipitacao"][df_filtrado["precipitacao"] > 1]
        if not valores_chuva.empty:
            fig_hist = px.histogram(
                valores_chuva, 
                nbins=40, 
                labels={'value': 'Precipitação Horária (mm)', 'count': 'Frequência'},
                template="plotly_white"
            )
            fig_hist.update_traces(marker_color='#009E73', marker_line_color='black', marker_line_width=0.5)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Nenhuma hora com chuva > 1 mm no período selecionado.")
# -----------------------------
# Aba 2: Máximas Anuais
# -----------------------------
with tab2:
    st.subheader("Análise das Máximas Anuais por Duração")
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
            labels={'Ano': 'Ano', 'Precipitação Máxima (mm)': f'Máxima em {duracao_max}h (mm)'},
            template="plotly_white"
        )
        fig_maximas.update_traces(marker_color='#56B4E9')
        st.plotly_chart(fig_maximas, use_container_width=True)
        
        with st.expander("Ver tabela de dados das máximas anuais"):
            st.dataframe(df_maximas.set_index("Ano"), use_container_width=True)

# -----------------------------
# Aba 3: Curvas IDF
# -----------------------------
with tab3:
    st.subheader("Geração de Curvas Intensidade-Duração-Frequência (IDF)")
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
        # Armazena resultados no estado da sessão para o PDF
        st.session_state['df_idf'] = df_idf
        #st.session_state['duracao_idf'] = duracao_idf
        st.session_state['params_gumbel'] = params_gumbel
        st.session_state['params_lp3'] = params_lp3
        
        # Gráfico IDF com Plotly
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
            title=f"Precipitação Máxima Estimada vs. Período de Retorno (Duração: {duracao_idf}h)",
            xaxis_title="Período de Retorno (anos)",
            yaxis_title="Precipitação Máxima (mm)",
            xaxis_type="log",
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_idf, use_container_width=True)
        
        # Salva a figura para o PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig_idf.write_image(tmpfile.name, scale=2)
            st.session_state['grafico_path'] = tmpfile.name
        
        # Apresentação dos resultados em colunas
        col_tabela, col_stats = st.columns([3, 2])
        with col_tabela:
            st.markdown("##### Tabela de Precipitações e Intensidades")
            st.dataframe(df_idf.style.format("{:.2f}"))
            
            csv_data = df_idf.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar Tabela (CSV)",
                data=csv_data,
                file_name=f"curvas_IDF_{duracao_idf}h.csv",
                mime="text/csv",
            )

        with col_stats:
            st.markdown("##### Resumo do Ajuste Estatístico")
            st.metric("Ajuste Gumbel (μ / β)", f"{params_gumbel['mu']:.2f} mm / {params_gumbel['beta']:.2f} mm")
            st.metric("Teste K-S (p-valor)", f"{params_gumbel['ks_p']:.3f}", 
                      "Aceito" if params_gumbel['ks_p'] > 0.05 else "Rejeitado (α=5%)")
        
        with st.expander("Ver detalhes dos parâmetros e da série de máximas"):
            st.write("**Parâmetros Gumbel:**", params_gumbel)
            st.write("**Parâmetros Log-Pearson III:**", params_lp3)
            st.write(f"**Série de Máximas Anuais ({duracao_idf}h) utilizada:**")
            st.dataframe(serie_maximas)

# -----------------------------
# Aba 4: Relatório PDF
# -----------------------------
with tab4:
    st.subheader("Geração de Relatório em PDF")
    st.markdown("Após calcular as curvas na aba **'📉 Curvas IDF'**, você pode gerar um relatório consolidado.")

    if 'df_idf' in st.session_state:
        st.info(f"Pronto para gerar o relatório para a duração de **{st.session_state['duracao_idf']} horas**.")
        
        if st.button("📄 Gerar e Baixar PDF"):
            with st.spinner("Montando seu relatório..."):
                pdf_obj = generate_professional_pdf(
                    st.session_state['df_idf'],
                    st.session_state['grafico_path'],
                    st.session_state['duracao_idf'],
                    st.session_state['params_gumbel'],
                    st.session_state['params_lp3']
                )
                
                # Salva o PDF em um arquivo temporário para download
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    pdf_obj.output(tmp.name)
                    with open(tmp.name, "rb") as f:
                        st.download_button(
                            label="📥 Clique aqui para baixar o Relatório PDF",
                            data=f.read(),
                            file_name=f"Relatorio_IDF_{st.session_state['duracao_idf']}h.pdf",
                            mime="application/pdf"
                        )
                # Limpa arquivos temporários
                os.remove(st.session_state['grafico_path'])
                
    else:
        st.warning("Por favor, gere uma curva na aba '📉 Curvas IDF' primeiro.")
