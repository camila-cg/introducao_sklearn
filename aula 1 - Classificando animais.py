#Projeto 1: Classificando animais

from sklearn.svm import LinearSVC

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

# 1 => porco, 0 => cachorro
dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
classes = [1,1,1,0,0,0]

#Criando modelo
model = LinearSVC()
model.fit(dados, classes)

#Testando o modelo
animal_misterioso = [1,1,1]
model.predict([animal_misterioso])

misterio1 = [1,1,1]
misterio2 = [1,1,0]
misterio3 = [0,1,1]  #porco q faz auau -> caso estranho

testes = [misterio1, misterio2, misterio3]
previsoes = model.predict(testes) #resultado do modelo
testes_classes = [0, 1, 1] #resultado esperado

#Exibindo taxa de acerto
corretos = (previsoes == testes_classes).sum()
total = len(testes)
taxa_de_acerto = corretos/total
print("Taxa de acerto ", taxa_de_acerto*100)