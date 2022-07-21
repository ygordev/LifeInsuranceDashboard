import streamlit as st
import pandas as pd
import numpy as np
from pycaret.regression import *

modelo = load_model('YgorML_LifeInsurance_RandomForest_21JUL2022')
variaveis = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

st.title('Previsão de Valor de Seguro de Vida (Regressão com PyCaret)')

# Sidebar
st.sidebar.header('Parâmetros')
age = st.sidebar.number_input("Escolha a sua idade", 8, 90)

sex = st.sidebar.selectbox("Escolha o sexo", ['Masculino', 'Feminino'])
if sex == "Masculino":
    sex = "male"
else:
    sex = "female"

bmi = st.sidebar.slider("Insira seu IMC", min_value=1.0, max_value=90.0, step=0.5)
bmi = float(bmi)

children = st.sidebar.number_input("Possui filhos", 0, 4)

smoker = st.sidebar.checkbox("Fumante?")
if smoker:
    smoker = "yes"
else:
    smoker = "no"

region = st.sidebar.selectbox("Escolha a Região (US)", ['Southwest', 'Southeast', 'Northwest', 'Northeast'])
region = region.lower()

dadosPrever = {
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
}
dfPrever = pd.DataFrame(dadosPrever)
st.write("ℹ️ Dados informados para a Previsão do Valor do Seguro:")
st.markdown(f"**Idade:** {age}")
st.markdown(f"**Sexo:** {sex}")
st.markdown(f"**IMC:** {bmi}")
st.markdown(f"**Qtd. Filhos:** {children}")
st.markdown(f"**Fumante?** {smoker}")
st.markdown(f"**Região:** {region}")

valor = predict_model(modelo, data=dfPrever)
st.markdown("**Valor Previsto ($):**")
st.write(float(valor['Label'][0]))