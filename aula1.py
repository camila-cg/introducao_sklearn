#Projeto 1: Classificando animais

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# features => atributos (1 sim, 0 não)
# pelo longo? 
# perna curta?
# faz auau?
porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

#modelo(dados) = classe --> f(x) = y

# 1 => porco, 0 => cachorro
treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3] #features
treino_y = [1,1,1,0,0,0]  #labels

#Criando modelo
model = LinearSVC()
model.fit(treino_x, treino_y)

#Testando o modelo
misterio1 = [1,1,1]
misterio2 = [1,1,0]
misterio3 = [0,1,1]  #porco q faz auau -> caso estranho

teste_x = [misterio1, misterio2, misterio3] #features
teste_y = [0, 1, 1] #labels
previsoes = model.predict(teste_x) #resultado do modelo

#Exibindo taxa de acerto
taxa_de_acerto = accuracy_score(teste_y, previsoes)
print("Taxa de acerto %.2f" %(taxa_de_acerto * 100))
