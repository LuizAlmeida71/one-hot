import streamlit as st
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

st.set_page_config(page_title="One-Hot Encoder (visual)", layout="wide")

def tokenize(text):
    # remove pontuação básica e separa por espaços
    text = text.lower()
    # substituir tudo que não for letra/número/espaço por espaço
    text = re.sub(r'[^0-9a-záàâãéèêíïóôõöúçñ ]+', ' ', text, flags=re.IGNORECASE)
    tokens = [t for t in text.split() if t != ""]
    return tokens

def one_hot_df(sentence):
    tokens = tokenize(sentence)
    if len(tokens) == 0:
        # DataFrame vazio consistente
        return pd.DataFrame(), []
    vocab = sorted(set(tokens))
    idx_map = {w:i for i,w in enumerate(vocab)}
    mat = np.zeros((len(tokens), len(vocab)), dtype=int)
    for i, w in enumerate(tokens):
        mat[i, idx_map[w]] = 1
    df = pd.DataFrame(mat, index=[f"{i}: {t}" for i,t in enumerate(tokens)], columns=vocab)
    return df, vocab

st.title("🧩 One-Hot Encoding — Visualizador")
st.write("Digite uma frase e veja a matriz one-hot com palavras como índices e colunas.")

text = st.text_area("Digite a frase aqui:", value="Eu amo aprender Python e Python é incrível, apesar de exigir muito da minha cognição.", height=140)

col1, col2 = st.columns([2,1])

with col1:
    if st.button("Gerar One-Hot"):
        df_onehot, vocab = one_hot_df(text)
        if df_onehot.empty:
            st.warning("Nenhuma palavra encontrada após tokenização.")
        else:
            st.subheader("📚 Vocabulário (colunas)")
            st.write(vocab)
            st.subheader("🧮 Matriz One-Hot (linhas = posição na frase)")
            # exibe DataFrame com scroll e largura responsiva
            st.dataframe(df_onehot, use_container_width=True)
            st.markdown("---")
            st.subheader("🔎 Mapa de calor da matriz One-Hot")
            # plot heatmap simples com matplotlib
            fig, ax = plt.subplots(figsize=(max(6, len(vocab)*0.4), max(3, len(df_onehot)*0.25)))
            im = ax.imshow(df_onehot.values, aspect='auto', interpolation='nearest')
            ax.set_yticks(range(len(df_onehot.index)))
            ax.set_yticklabels(df_onehot.index, fontsize=9)
            ax.set_xticks(range(len(df_onehot.columns)))
            ax.set_xticklabels(df_onehot.columns, rotation=90, fontsize=9)
            ax.set_xlabel("Vocabulário")
            ax.set_ylabel("Posição : palavra")
            plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.02)
            st.pyplot(fig)
with col2:
    st.subheader("Dicas")
    st.markdown("""
    - Linhas: cada token (posição na frase).  
    - Colunas: palavras únicas no vocabulário.  
    - `1` indica que a palavra da linha corresponde à coluna.  
    - A tokenização remove pontuação simples; ajuste `tokenize()` se quiser regras diferentes.
    """)
