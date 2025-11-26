import pandas as pd
import numpy as np
import re
import os
import random
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode

def carregar_dados(train_csv, valid_csv):
    df_treino = pd.read_csv(train_csv) # treino e teste em data frames
    df_teste = pd.read_csv(valid_csv)

    X_train = df_treino.drop(columns=["image_path"]).values # mantendo somente features
    X_valid = df_teste.drop(columns=["image_path"]).values

    # rotulo do nome do personagem no início do arquivo
    y_train = [re.match(r"([a-zA-Z]+)", os.path.basename(img)).group(1) for img in df_treino["image_path"]]
    y_valid = [re.match(r"([a-zA-Z]+)", os.path.basename(img)).group(1) for img in df_teste["image_path"]]

    # rotulos viram numeros; bart 0, homer 1 etc
    le = LabelEncoder()
    y_train_num = le.fit_transform(y_train) # rotulos de treino e teste
    y_valid_num = le.transform(y_valid)

    return X_train, y_train_num, X_valid, y_valid_num, le

def knn(X_train, y_train, X_valid, y_valid, cv_folds=10):
    print("k-NN")
    knn = KNeighborsClassifier(n_neighbors=5)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random.randint(0, 10000))
    
    f1_cv = cross_val_score(knn, X_train, y_train, cv=cv, scoring='f1_macro') 
    print(f"f1-score da cross validation: {f1_cv.mean()*100:.2f}%") # media do f1 score dos treinos
    
    knn.fit(X_train, y_train) # treino
    y_pred = knn.predict(X_valid) # predição
    
    f1 = f1_score(y_valid, y_pred, average='macro') 
    print(f"f1-score do teste de validação: {f1*100:.2f}%") # f1 score dos testes

    # matriz de confusão
    cm = confusion_matrix(y_valid, y_pred)  # linhas = labels verdadeiros, colunas = labels previstos

    # calcular percentual por linha p/ cada classe real
    matriz_confusao = []
    for i, row in enumerate(cm):
        total = row.sum() # total de exemplos
        if total == 0:
            percentual = [0]*len(row) # linha de zeros
        else:
            percentual = (row / total) * 100 # percentual por classe
        matriz_confusao.append(percentual)
    matriz_confusao = np.array(matriz_confusao) 

    print("matriz de confusão %:")
    labels = le.classes_ 
    print(pd.DataFrame(np.round(matriz_confusao,2), index=labels, columns=labels))

def svm(X_train, y_train, X_valid, y_valid, cv_folds=10):
    print("\nSVM")
    svm = SVC(kernel='linear', C=1.0)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random.randint(0, 10000))
    
    f1_cv = cross_val_score(svm, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"f1-score da cross validation: {f1_cv.mean()*100:.2f}%") 
    
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_valid)
    
    f1 = f1_score(y_valid, y_pred, average='macro')
    print(f"f1-score do teste de validação: {f1*100:.2f}%") 

    # matriz de confusão
    cm = confusion_matrix(y_valid, y_pred)

    # calcular percentual por linha p/ cada classe real
    matriz_confusao = []
    for i, row in enumerate(cm):
        total = row.sum() # total de exemplos
        if total == 0:
            percentual = [0]*len(row) # linha de zeros
        else:
            percentual = (row / total) * 100 # percentual por classe
        matriz_confusao.append(percentual)
    matriz_confusao = np.array(matriz_confusao) 

    print("matriz de confusão %:")
    labels = le.classes_ 
    print(pd.DataFrame(np.round(matriz_confusao,2), index=labels, columns=labels))

def random_forest(X_train, y_train, X_valid, y_valid, cv_folds=10):
    print("\nRandom Forest")

    rf = RandomForestClassifier(
        n_estimators=200, # 200 arvores
        max_depth=None,
        random_state=random.randint(0, 10000)
    )

    # cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random.randint(0, 10000))

    f1_cv = cross_val_score(rf, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"f1-score da cross validation: {f1_cv.mean()*100:.2f}%")

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_valid)

    f1 = f1_score(y_valid, y_pred, average='macro')
    print(f"f1-score do teste de validação: {f1*100:.2f}%")

    # matriz de confusão
    cm = confusion_matrix(y_valid, y_pred)

    # calcular percentual por linha p/ cada classe real
    matriz_confusao = []
    for i, row in enumerate(cm):
        total = row.sum() # total de exemplos
        if total == 0:
            percentual = [0]*len(row) # linha de zeros
        else:
            percentual = (row / total) * 100 # percentual por classe
        matriz_confusao.append(percentual)
    matriz_confusao = np.array(matriz_confusao) 

    print("matriz de confusão %:")
    labels = le.classes_
    print(pd.DataFrame(np.round(matriz_confusao, 2), index=labels, columns=labels))

def combinacao_estatica(X_train, y_train, X_valid, y_valid):
    print("\nensemble de 20 classificadores random forest")

    models = [] # modelos
    preds = [] # predição de cada modelo

    for i in range(20):
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=i
        )

        rf.fit(X_train, y_train)

        models.append(rf) # guarda modelo
        preds.append(rf.predict(X_valid)) # guarda predições do modelo

    # matriz 20 modelos × N amostras
    preds = np.array(preds)

    final_pred = mode(preds, axis=0, keepdims=False).mode# voto majoritário

    f1 = f1_score(y_valid, final_pred, average='macro')
    print(f"f1-score do ensemble: {f1*100:.2f}%")

    # matriz de confusão
    cm = confusion_matrix(y_valid, final_pred)

    # calcular percentual por linha p/ cada classe real
    matriz_confusao = []
    for i, row in enumerate(cm):
        total = row.sum() # total de exemplos
        if total == 0:
            percentual = [0]*len(row) # linha de zeros
        else:
            percentual = (row / total) * 100 # percentual por classe
        matriz_confusao.append(percentual)
    matriz_confusao = np.array(matriz_confusao) 

    print("matriz de confusão %:")
    labels = le.classes_
    print(pd.DataFrame(np.round(matriz_confusao, 2), index=labels, columns=labels))


if __name__ == "__main__":
    X_train, y_train, X_valid, y_valid, le = carregar_dados("features_train.csv", "features_valid.csv")
    
    knn(X_train, y_train, X_valid, y_valid)
    svm(X_train, y_train, X_valid, y_valid)
    random_forest(X_train, y_train, X_valid, y_valid)
    combinacao_estatica(X_train, y_train, X_valid, y_valid)