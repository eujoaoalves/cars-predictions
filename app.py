import os
import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

# 🛠️ Garante que o caminho do modelo seja correto
model_path = os.path.join(os.path.dirname(__file__), "modelo.pkl")

# 🛠️ Verifica se o modelo existe antes de tentar carregar
if not os.path.exists(model_path):
    st.error("Erro: Arquivo 'modelo.pkl' não encontrado. Verifique o caminho do arquivo no deploy.")
    st.stop()

# 🔄 Carrega o modelo
with open(model_path, "rb") as file:
    model = pk.load(file)

# 🛠️ Carregamento seguro do dataset
data_path = os.path.join(os.path.dirname(__file__), "base_de_dados_carro.csv")

if not os.path.exists(data_path):
    st.error("Erro: Arquivo 'base_de_dados_carro.csv' não encontrado.")
    st.stop()

carros = pd.read_csv(data_path)

# 🔧 Função para extrair a marca do carro
def nome_marca(marca):
    return marca.split(" ")[0].strip()

# 🔄 Renomeando colunas para português
colunas_traduzidas = {
    "name": "nome",
    "year": "ano",
    "selling_price": "preco_venda",
    "km_driven": "km_rodados",
    "fuel": "combustivel",
    "seller_type": "tipo_vendedor",
    "transmission": "transmissao",
    "owner": "proprietario",
    "mileage": "consumo",
    "engine": "motor",
    "max_power": "potencia_maxima",
    "seats": "assentos",
}

carros = carros.rename(columns=colunas_traduzidas)
carros["nome"] = carros["nome"].apply(nome_marca)

# 📌 Interface Streamlit
st.header("Predição de Preços de Carros")

# 🔽 Inputs do usuário
nome = st.selectbox("Marca", carros["nome"].unique())
ano = st.slider("Ano", 2000, 2024, value=2023)
km_rodados = st.slider("Kilômetros Rodados", 11, 200000, value=20000)
combustivel = st.selectbox("Combustível", carros["combustivel"].unique())
tipo_vendedor = st.selectbox("Tipo de Vendedor", carros["tipo_vendedor"].unique())
proprietarios = st.selectbox("Qtd Proprietários", carros["proprietario"].unique())
transmissao = st.selectbox("Transmissão", carros["transmissao"].unique())

consumo = st.slider("Consumo (km/L)", 11, 40)
motor = st.slider("Motor (cc)", 700, 5000)
potencia_maxima = st.slider("Potência Máxima (hp)", 0, 200)
assentos = st.slider("Assentos", 2, 10)

if st.button("Prever"):
    input_data = pd.DataFrame(
        [[nome, ano, km_rodados, combustivel, tipo_vendedor, proprietarios, transmissao, consumo, motor, potencia_maxima, assentos]],
        columns=["nome", "ano", "km_rodados", "combustivel", "tipo_vendedor", "proprietario", "transmissao", "consumo", "motor", "potencia_maxima", "assentos"],
    )

    # 🔄 Substituição de valores categóricos por números
    marca_mapping = {marca: i+1 for i, marca in enumerate(carros["nome"].unique())}
    input_data["nome"] = input_data["nome"].map(marca_mapping)

    input_data["transmissao"].replace(["Manual", "Automatic"], [1, 2], inplace=True)
    input_data["proprietario"].replace(["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"], [1, 2, 3, 4, 5], inplace=True)
    input_data["combustivel"].replace(["Diesel", "Petrol", "LPG", "CNG"], [1, 2, 3, 4], inplace=True)
    input_data["tipo_vendedor"].replace(["Individual", "Dealer", "Trustmark Dealer"], [1, 2, 3], inplace=True)

    # 📌 Ajusta a ordem das colunas conforme o modelo
    input_data = input_data[model.feature_names_in_]

    # 🛠️ Converte para float, se necessário
    input_data = input_data.astype(float)

    # 🔮 Predição
    preco_do_carro = model.predict(input_data)[0]
    st.markdown(f"💰 **Preço Estimado do Carro:** ${preco_do_carro:.2f} dólares")
