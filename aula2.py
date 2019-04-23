#Projeto 2: Classificando compras

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#Lendo dados de uma fonte externa
uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
dados = pd.read_csv(uri)

#renomeando colunas
mapa = {
    "home" : "principal",
    "how_it_works" : "como_funciona",
    "contact" : "contato",
    "bought" : "comprou"
}
dados = dados.rename(columns = mapa)

#Exibe os 5 primeiros registros
print(dados.head())

#modelo(dados) = classe --> f(x) = y
x = dados[["principal","como_funciona","contato"]]
y = dados["comprou"]

# Informa a qtde de linhas e colunas do dado
print(dados.shape)

#Separando a base
SEED = 20
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state = SEED, test_size =0.25, stratify = y)
print(treino_x.shape)
print(teste_x.shape)

#analisando
print(treino_y.value_counts())
print(teste_y.value_counts())

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

#Criando modelo
modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x) #resultado do modelo

#Exibindo acuracia do modelo
acuracia = accuracy_score(teste_y, previsoes)
print("Taxa de acerto %.2f" %(acuracia * 100))
