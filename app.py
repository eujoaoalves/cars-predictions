import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

model = pk.load(open('modelo.pkl', 'rb'))
carros = pd.read_csv('base_de_dados_carro.csv')

def nome_marca(marca):
    nome =  marca.split(' ')[0]
    return nome.strip()
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
    "seats": "assentos"
}
carros = carros.rename(columns=colunas_traduzidas)
carros["nome"] = carros["nome"].apply(nome_marca)

st.header( 'Predição preços de carros')
nome = st.selectbox('Marca', carros['nome'].unique())
ano = st.slider('Ano', 2000,2024, value=2023)
km_rodados =  st.slider('Kilometros Rodados', 11,200000, value=20000)
combustivel = st.selectbox('combustivel',carros["combustivel"].unique())
tipo_vendedor = st.selectbox('Tipo de Vendedor',carros["tipo_vendedor"].unique())
proprietarios = st.selectbox('Qtd Proprietários',carros["proprietario"].unique())
transmissao= st.selectbox('Transmissão',carros["transmissao"].unique())

consumo = st.slider('Consumo',11,40)
motor = st.slider('Motor',700,5000)
potencia_maxima = st.slider('Potência Máxima',0,200)
assentos = st.slider('Assentos',5,10)

if st.button('Prever'):
    input_data = pd.DataFrame(
            [[nome, ano, km_rodados, combustivel, tipo_vendedor, proprietarios, transmissao, consumo, motor, potencia_maxima, assentos]],
            columns=['nome', 'ano', 'km_rodados', 'combustivel', 'tipo_vendedor', 'proprietario', 'transmissao', 'consumo', 'motor', 'potencia_maxima', 'assentos']
    )

    input_data['nome'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,27,28,29,30,31],
       inplace=True)
    
    input_data['transmissao'].replace(['Manual', 'Automatic'], [1,2], inplace=True)
    input_data["proprietario"].replace(['First Owner', 'Second Owner', 'Third Owner',
       'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5], inplace=True)
    input_data["combustivel"].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    input_data["tipo_vendedor"].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)

    # 🛠️ Ajusta a ordem das colunas para ser a mesma usada no treinamento do modelo
    input_data = input_data[model.feature_names_in_]

    # 🛠️ Converte os dados para float, caso o modelo tenha sido treinado com esse tipo de dado
    input_data = input_data.astype(float)

    st.write("Dados formatados para o modelo:")
    st.write(input_data)

    preco_do_carro = model.predict(input_data)
    preco_do_carro_reais = preco_do_carro[0]
    st.markdown(f'O preço do carro é: ${preco_do_carro_reais:.2f} dólares')