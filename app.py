import pandas as pd
import streamlit as st
from pycaret.regression import *
import numpy as np
 

# loading the trained model.
model = load_model('modelo')

# carregando uma amostra dos dados.
dataset = pd.read_csv('students.csv') 
#classifier = pickle.load(pickle_in)

# título
st.title("Predição de notas de matemática")

# subtítulo
st.markdown("Este é um Data App utilizado para exibir a solução de Machine Learning para o problema de predição de notas de matemática.")
st.sidebar.subheader("Defina os atributos do aluno para a predição da nota de matemática")

# mapeando dados do usuário para cada atributo
nota_leitura = st.sidebar.number_input("Nota de Leitura")
nota_escrita = st.sidebar.number_input("Nota de Escrita")

genero = st.sidebar.selectbox("Gênero do Aluno",("Feminino","Masculino"))
etinia = st.sidebar.selectbox("Raça/Etinia",("Grupo A","Grupo B","Grupo C","Grupo D","Grupo E"))
educacao_pais = st.sidebar.selectbox("Grau de Escolaridade",("bachelor's degree","some college","master's degree","associate's degree","high school", "some high school"))
curso_preparacao = st.sidebar.selectbox("Curso Preparatório para Teste",("Nenhum","Completo"))
almoco = st.sidebar.selectbox("Tipo de Almoço",("Gratuito/Reduzido","Padrão"))

# transformando o dado de entrada em valor binário
female = 1 if genero == "Feminino" else 0
male = 1 if genero == "Masculino" else 0

group_A = 1 if etinia == "Grupo A" else 0
group_B = 1 if etinia == "Grupo B" else 0
group_C = 1 if etinia == "Grupo C" else 0
group_D = 1 if etinia == "Grupo D" else 0
group_E = 1 if etinia == "Grupo E" else 0

associates_degree = 1 if educacao_pais == "Diploma de técnico" else 0
bachelors_degree = 1 if educacao_pais == "Graduação" else 0
high_school = 1 if educacao_pais == "Ensino médio" else 0
masters_degree = 1 if educacao_pais == "Mestrado" else 0
some_college = 1 if educacao_pais == "Ensino superior incompleto" else 0
some_high_school = 1 if educacao_pais == "Ensino médio incompleto" else 0


completed = 1 if curso_preparacao == "Completo" else 0
none = 1 if curso_preparacao == "Nenhum" else 0

freereduced = 1 if almoco == "Gratuito/Reduzido" else 0
standard = 1 if almoco == "Padrão" else 0


# inserindo um botão na tela
btn_predict = st.sidebar.button("Realizar Predição")

# verifica se o botão foi acionado
if btn_predict:
    data_teste = pd.DataFrame()

    data_teste["reading_score"] = [nota_leitura]
    data_teste["writing_score"] = [nota_escrita]    
    data_teste["female"] = [female]
    data_teste["male"] = [male]	
    data_teste["group_A"] = [group_A]
    data_teste["group_B"] = [group_B]
    data_teste["group_C"] = [group_C]
    data_teste["group_D"] =	[group_D]
    data_teste["group_E"] =	[group_E]
    data_teste["associates_degree"] = [associates_degree]
    data_teste["bachelors_degree"] = [bachelors_degree]
    data_teste["high_school"] = [high_school]
    data_teste["masters_degree"] = [masters_degree]
    data_teste["some_college"] = [some_college]
    data_teste["some_high_school"] = [some_high_school]
    data_teste["freereduced"] = [freereduced]
    data_teste["standard"] = [standard]
    data_teste["completed"] = [completed]
    data_teste["none"] = [none]


    #imprime os dados de teste    
    print(data_teste)

    #realiza a predição
    result = model.predict(data_teste)
    
    st.subheader("Nota de matematica predita:")
    result = (round(result[0],2))
    
    st.write(result)
   