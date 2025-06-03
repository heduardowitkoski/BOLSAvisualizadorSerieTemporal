import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import customtkinter as ctk
from tkinter import filedialog, messagebox

# =========================
# VARIÁVEIS GLOBAIS
# =========================

df_global = None  # DataFrame carregado
variaveis_meteorologicas = {
    "Temperatura": "Temp. Ins. (C)",
    "Chuva": "Chuva (mm)"
}

# =========================
# CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# =========================

def carregar_dados(caminho):
    df = pd.read_csv(
        caminho,
        sep=';', encoding='utf-8', decimal=',',
        na_values=['', ' '], quotechar='"'
    )
    df.columns = df.columns.str.strip()
    df['Hora (UTC)'] = df['Hora (UTC)'].astype(str).str.strip().str.zfill(4)
    df['DataHora'] = pd.to_datetime(
        df['Data'].astype(str) + ' ' +
        df['Hora (UTC)'].str[:2] + ':' +
        df['Hora (UTC)'].str[2:],
        format='%d/%m/%Y %H:%M',
        errors='coerce'
    )
    df = df.dropna(subset=['DataHora'])
    df['AnoMes'] = df['DataHora'].dt.to_period('M')
    return df

# =========================
# GERAÇÃO DO GRÁFICO
# =========================

def gerar_figure(df, variavel, tipo):
    if variavel not in variaveis_meteorologicas:
        messagebox.showerror("Erro", f"Variável '{variavel}' inválida.")
        return None
    col = variaveis_meteorologicas[variavel]
    if col not in df.columns:
        messagebox.showerror("Erro", f"Coluna '{col}' ausente no arquivo.")
        return None
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df_filtrado = df.dropna(subset=[col])
    media_mensal = df_filtrado.groupby('AnoMes')[col].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    if tipo == "Linhas":
        media_mensal.plot(ax=ax, marker='o')
    else:
        media_mensal.plot(kind='bar', ax=ax)

    ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(media_mensal))))
    labels, ultimo_mes = [], None
    for periodo in media_mensal.index:
        mes = periodo.strftime('%b').capitalize()
        labels.append(mes if mes != ultimo_mes else '')
        ultimo_mes = mes
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(f"Média mensal de {variavel}")
    ax.set_xlabel("Mês")
    ax.set_ylabel(f"{variavel} (média)")
    ax.grid(True)
    fig.tight_layout()
    return fig

# =========================
# PLOTAGEM DO GRÁFICO
# =========================

def plotar_grafico_tk(df, variavel, tipo):
    fig = gerar_figure(df, variavel, tipo)
    if not fig:
        return
    if hasattr(plotar_grafico_tk, 'canvas') and plotar_grafico_tk.canvas:
        plotar_grafico_tk.canvas.get_tk_widget().destroy()
    canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    plotar_grafico_tk.canvas = canvas

# =========================
# INTERAÇÕES COM O USUÁRIO
# =========================

def carregar_csv():
    global df_global
    caminho = filedialog.askopenfilename(filetypes=[("Arquivos CSV", "*.csv")])
    if not caminho:
        return
    try:
        df_global = carregar_dados(caminho)
        messagebox.showinfo("Sucesso", "Dados carregados com sucesso.")
    except Exception as e:
        messagebox.showerror("Erro ao carregar CSV", str(e))

def executar_plot():
    if df_global is None:
        messagebox.showwarning("Aviso", "Carregue um arquivo CSV primeiro.")
        return
    variavel = combobox_variavel.get()
    tipo = combobox_grafico.get()
    plotar_grafico_tk(df_global, variavel, tipo)

def mostrar_tela_principal():
    frame_inicio.pack_forget()
    frame_principal.pack(padx=10, pady=10, fill="x")
    frame_grafico.pack(fill="both", expand=True, padx=10, pady=10)

# =========================
# INTERFACE GRÁFICA
# =========================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

janela = ctk.CTk()
janela.title("Visualizador de Séries Temporais")
janela.geometry("800x600")

# --------- Tela inicial ---------
frame_inicio = ctk.CTkFrame(janela)
frame_inicio.pack(fill="both", expand=True)

titulo = ctk.CTkLabel(frame_inicio, text="Visualizador de Séries Temporais", font=ctk.CTkFont(size=28, weight="bold"))
titulo.pack(pady=60)

botao_iniciar = ctk.CTkButton(frame_inicio, text="Iniciar", command=mostrar_tela_principal, width=200, height=40)
botao_iniciar.pack(pady=20)

# --------- Tela principal ---------
frame_principal = ctk.CTkFrame(janela)

botao_carregar = ctk.CTkButton(frame_principal, text="Carregar CSV", command=carregar_csv)
botao_carregar.grid(row=0, column=0, padx=5, pady=5)

combobox_variavel = ctk.CTkComboBox(frame_principal, values=list(variaveis_meteorologicas.keys()), width=150)
combobox_variavel.grid(row=0, column=1, padx=5, pady=5)
combobox_variavel.set("Temperatura")

combobox_grafico = ctk.CTkComboBox(frame_principal, values=["Linhas", "Barras"], width=100)
combobox_grafico.grid(row=0, column=2, padx=5, pady=5)
combobox_grafico.set("Linhas")

botao_gerar = ctk.CTkButton(frame_principal, text="Gerar Gráfico", command=executar_plot)
botao_gerar.grid(row=0, column=3, padx=5, pady=5)

frame_grafico = ctk.CTkFrame(janela)

# =========================
# EXECUTA A JANELA
# =========================

janela.mainloop()
