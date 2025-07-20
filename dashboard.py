import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r, pearson3, kstest, anderson
from fpdf import FPDF
import tempfile
import os

# -----------------------------
# Configuração da página
# -----------------------------
st.set_page_config(page_title="Dashboard de Chuvas", layout="wide")
st.title("Sistema de Análise Pluviométrica (SAP-IDF)")

st.markdown("""
Este painel permite:
- Enviar dados horários de precipitação,
- Selecionar intervalo de anos,
- Calcular máximas anuais,
- Ajustar distribuições (Gumbel e Log‑Pearson III),
- Testar aderência estatística,
- Gerar curvas IDF e relatório PDF.
""")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Configurações")
uploaded_file = st.sidebar.file_uploader("📂 Envie o CSV de chuva horária", type=["csv"])

# -----------------------------
# Função auxiliar: gerar relatório PDF
# -----------------------------
def gerar_pdf(df_idf, grafico_path, duracao):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(0, 10, "Relatório de Curvas IDF", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(8)
    pdf.cell(0, 10, f"Duração analisada: {duracao}h", ln=True)
    pdf.ln(5)
    pdf.set_font("Courier", size=10)
    for linha in df_idf.to_string(index=False).split("\n"):
        pdf.multi_cell(0, 6, linha)
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(0, 10, "Gráfico IDF", ln=True, align="C")
    pdf.image(grafico_path, w=170)
    return pdf

# -----------------------------
# Processamento
# -----------------------------
if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file, sep=None, engine='python')
        df_raw.columns = [col.strip().lower() for col in df_raw.columns]

        if "data" in df_raw.columns and "hora" in df_raw.columns and "precipitacao" in df_raw.columns:
            df_raw["precipitacao"] = df_raw["precipitacao"].astype(str).str.replace(",", ".", regex=False).astype(float)
            df_raw["hora"] = df_raw["hora"].astype(str).str.zfill(4)
            df_raw["datahora"] = pd.to_datetime(
                df_raw["data"].astype(str) + " " + df_raw["hora"].str[:2] + ":" + df_raw["hora"].str[2:],
                format="%Y-%m-%d %H:%M",
                errors="coerce"
            )
            df_raw = df_raw.dropna(subset=["datahora"])
            df = df_raw[["datahora", "precipitacao"]]
        elif "datahora" in df_raw.columns and "precipitacao" in df_raw.columns:
            df_raw["precipitacao"] = df_raw["precipitacao"].astype(str).str.replace(",", ".", regex=False).astype(float)
            df_raw["datahora"] = pd.to_datetime(df_raw["datahora"], errors="coerce")
            df_raw = df_raw.dropna(subset=["datahora"])
            df = df_raw[["datahora", "precipitacao"]]
        else:
            st.error("Formato de CSV não reconhecido. Verifique se contém as colunas 'data' e 'hora' ou 'datahora' e 'precipitacao'.")
            st.stop()

        df = df.sort_values("datahora").set_index("datahora")

        st.success(f"✅ Arquivo carregado: {uploaded_file.name}")

        # Intervalo de anos
        anos_disponiveis = sorted(df.index.year.unique())
        ano_min = st.sidebar.selectbox("Ano inicial", anos_disponiveis, index=0)
        ano_max = st.sidebar.selectbox("Ano final", anos_disponiveis, index=len(anos_disponiveis)-1)
        df = df[(df.index.year >= ano_min) & (df.index.year <= ano_max)]
        st.markdown(f"**Intervalo selecionado:** {ano_min} – {ano_max} ({len(df)} registros)")

        # Abas principais
        aba = st.tabs(["📈 Séries Temporais", "📊 Máximas e IDF", "📄 Relatório PDF"])

        # Aba 1 – Séries Temporais
        with aba[0]:
            st.subheader("Série Temporal da Precipitação Horária")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(df.index, df["precipitacao"], color="blue", linewidth=0.8)
            ax.set_xlabel("Data")
            ax.set_ylabel("Precipitação (mm)")
            ax.set_title("Série Horária de Precipitação")
            st.pyplot(fig)
            st.markdown("Prévia dos dados:")
            st.dataframe(df.head(50))

        # Aba 2 – Máximas e IDF
        with aba[1]:
            st.subheader("Configurações para cálculo de máximas e IDF")
            janelas = st.multiselect("Selecione janelas (horas):", [1,2,3,6,12,24], default=[1,3,6])

            acumulados = pd.DataFrame(index=df.index)
            for j in janelas:
                acumulados[f"acc_{j}h"] = df["precipitacao"].rolling(window=j, min_periods=1).sum()
            acumulados["ano"] = acumulados.index.year

            max_anuais = {j: acumulados.groupby("ano")[f"acc_{j}h"].max() for j in janelas}
            df_max = pd.DataFrame(max_anuais)
            df_max.index.name = "Ano"

            st.markdown("**Máximas anuais por janela (tabela):**")
            st.dataframe(df_max.head(50))
            st.download_button("📥 Baixar máximas anuais (CSV)", df_max.to_csv().encode("utf-8"), "maximos_anuais.csv")

            duracao = st.selectbox("Escolha duração para curva IDF:", janelas)
            serie = max_anuais[duracao].dropna()

            if len(serie) > 3:
                st.markdown(f"### Ajustes estatísticos para {duracao}h")

                mu_g, beta_g = gumbel_r.fit(serie.values)
                st.write(f"Gumbel: μ={mu_g:.2f}, β={beta_g:.2f}")

                dados_log = np.log10(serie.values)
                skew = pd.Series(dados_log).skew()
                mean_log = np.mean(dados_log)
                std_log = np.std(dados_log, ddof=1)
                lp3_dist = pearson3(skew)
                st.write(f"Log‑Pearson III: média log={mean_log:.3f}, desvio log={std_log:.3f}, assimetria={skew:.3f}")

                ks_stat, ks_p = kstest(serie.values, 'gumbel_r', args=(mu_g, beta_g))
                ad_result = anderson((serie.values - mu_g)/beta_g, dist='gumbel')
                st.markdown("**Testes de Aderência (Gumbel):**")
                st.write(f"Kolmogorov–Smirnov: estatística={ks_stat:.4f}, p={ks_p:.4f}")
                st.write(f"Anderson–Darling: estatística={ad_result.statistic:.4f}, valores críticos={ad_result.critical_values}")

                trs = np.array([2,5,10,25,50,100])
                intensidades_gumbel = []
                intensidades_lp3 = []
                for tr in trs:
                    F = 1 - 1/tr
                    x_g = mu_g - beta_g * np.log(-np.log(F))
                    intensidades_gumbel.append(x_g)
                    z = lp3_dist.ppf(F)
                    x_lp3 = 10 ** (mean_log + std_log * z)
                    intensidades_lp3.append(x_lp3)
                df_idf = pd.DataFrame({
                    "TR (anos)": trs,
                    f"Gumbel_{duracao}h (mm)": intensidades_gumbel,
                    f"LP3_{duracao}h (mm)": intensidades_lp3
                })
                st.markdown("**Curvas IDF calculadas:**")
                st.dataframe(df_idf)
                st.download_button("📥 Baixar curvas IDF (CSV)", df_idf.to_csv(index=False).encode("utf-8"), f"curvas_IDF_{duracao}h.csv")

                fig_idf, ax_idf = plt.subplots()
                ax_idf.plot(trs, intensidades_gumbel, marker="o", label="Gumbel")
                ax_idf.plot(trs, intensidades_lp3, marker="s", label="Log‑Pearson III")
                ax_idf.set_xscale("log")
                ax_idf.set_xlabel("Período de Retorno (anos)")
                ax_idf.set_ylabel("Precipitação Máxima (mm)")
                ax_idf.set_title(f"Curvas IDF – Duração {duracao}h")
                ax_idf.grid(True, which="both", linestyle="--")
                ax_idf.legend()
                st.pyplot(fig_idf)

                grafico_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig_idf.savefig(grafico_temp.name, dpi=150)
                st.session_state["df_idf"] = df_idf
                st.session_state["grafico_path"] = grafico_temp.name
                st.session_state["duracao"] = duracao
            else:
                st.warning("Série insuficiente para ajuste estatístico.")

        # Aba 3 – Relatório PDF
        with aba[2]:
            st.subheader("Gerar relatório PDF")
            if "df_idf" in st.session_state:
                if st.button("📄 Gerar PDF"):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        pdf_path = os.path.join(tmpdir, "relatorio.pdf")
                        pdf = gerar_pdf(st.session_state["df_idf"], st.session_state["grafico_path"], st.session_state["duracao"])
                        pdf.output(pdf_path)
                        with open(pdf_path, "rb") as f:
                            st.download_button("📥 Baixar Relatório PDF", f, file_name="relatorio.pdf")
            else:
                st.info("Calcule primeiro as curvas IDF na aba anterior para gerar o relatório.")
    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {e}")
else:
    st.info("Envie um arquivo CSV para começar.")
