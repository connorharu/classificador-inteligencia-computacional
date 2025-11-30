import pandas as pd
import numpy as np
import re
import os
import random
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode
from sklearn.neural_network import MLPClassifier

def carregar_dados(train_csv, valid_csv):
    df_treino = pd.read_csv(train_csv) # treino e teste em data frames
    df_teste = pd.read_csv(valid_csv)

    X_train = df_treino.drop(columns=["image_path"]).values # mantendo somente features
    X_valid = df_teste.drop(columns=["image_path"]).values

    # rotulo do nome do personagem no início do arquivo
    y_train = [img.split("\\")[-1][:-7] for img in df_treino["image_path"]]
    y_valid = [img.split("\\")[-1][:-7] for img in df_teste["image_path"]]

    # rotulos viram numeros; bart 0, homer 1 etc
    le = LabelEncoder()
    y_train_num = le.fit_transform(y_train) # rotulos de treino e teste
    y_valid_num = le.transform(y_valid)

    return X_train, y_train_num, X_valid, y_valid_num, le

def metricas(y_valid, y_pred):
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


def arvoreDecisao(X_train, y_train, X_valid, y_valid, cv_folds=10):
    print("Árvore de Decisão")
    tree = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=8,
        min_samples_split=2,
        random_state=random.randint(0, 10000)
    )
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random.randint(0, 10000)
    )

    f1_cv = cross_val_score(tree, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"f1-score da cross validation: {f1_cv.mean() * 100:.2f}%")

    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_valid)

    metricas(y_valid, y_pred)

def mlp_classificador(X_train, y_train, X_valid, y_valid, cv_folds=10):
    print("MLP (Multi-Layer Perceptron)")

    mlp = MLPClassifier(
        hidden_layer_sizes=[100,50],   
        activation='relu',              
        solver='adam',                 
        max_iter=100000,                
        random_state=random.randint(0, 10000)
    )

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random.randint(0, 10000)
    )

    f1_cv = cross_val_score(mlp, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"f1-score da cross validation: {f1_cv.mean() * 100:.2f}%")

    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_valid)

    metricas(y_valid, y_pred)

if __name__ == "__main__":
    X_train, y_train, X_valid, y_valid, le = carregar_dados("features_train.csv", "features_valid.csv")
    
    arvoreDecisao(X_train, y_train, X_valid, y_valid)
    mlp_classificador(X_train, y_train, X_valid, y_valid)