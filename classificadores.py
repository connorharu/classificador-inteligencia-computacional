import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import re
import os
import random

def carregar_dados(train_csv, valid_csv):
    df_train = pd.read_csv(train_csv) # treino e teste em data frames
    df_valid = pd.read_csv(valid_csv)

    X_train = df_train.drop(columns=["image_path"]).values # mantendo somente features
    X_valid = df_valid.drop(columns=["image_path"]).values

    # rotulo do nome do personagem no início do arquivo
    y_train = [re.match(r"([a-zA-Z]+)", os.path.basename(img)).group(1) for img in df_train["image_path"]]
    y_valid = [re.match(r"([a-zA-Z]+)", os.path.basename(img)).group(1) for img in df_valid["image_path"]]

    # rotulos viram numeros; bart 0, homer 1 etc
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train) # rotulos de treino e teste
    y_valid_enc = le.transform(y_valid)

    return X_train, y_train_enc, X_valid, y_valid_enc, le

def knn(X_train, y_train, X_valid, y_valid, cv_folds=10):
    print("k-NN")
    knn = KNeighborsClassifier(n_neighbors=5)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random.randint(0, 10000))
    
    scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='f1_macro') 
    print(f"f1-score da cross validation: {scores.mean()*100:.2f}%") # media do f1 score dos treinos
    
    knn.fit(X_train, y_train) # treino
    y_pred = knn.predict(X_valid) # predição
    
    f1 = f1_score(y_valid, y_pred, average='macro') 
    print(f"f1-score do teste de validação: {f1*100:.2f}%") # f1 score dos testes

    # matriz de confusão
    cm = confusion_matrix(y_valid, y_pred)  # linhas = labels verdadeiros, colunas = labels previstos

    # calcular percentual por linha p/ cada classe real
    cm_percent = []
    for i, row in enumerate(cm):
        row_total = row.sum() # total de exemplos
        if row_total == 0:
            row_percent = [0]*len(row) # linha de zeros
        else:
            row_percent = (row / row_total) * 100 # percentual por classe
        cm_percent.append(row_percent)
    cm_percent = np.array(cm_percent) 

    print("matriz de confusão %:")
    labels = le.classes_ 
    print(pd.DataFrame(np.round(cm_percent,2), index=labels, columns=labels))

def svm(X_train, y_train, X_valid, y_valid, cv_folds=10):
    print("\nSVM")
    svm_model = SVC(kernel='linear', C=1.0)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random.randint(0, 10000))
    
    scores = cross_val_score(svm_model, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"f1-score da cross validation: {scores.mean()*100:.2f}%") 
    
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_valid)
    
    f1 = f1_score(y_valid, y_pred, average='macro')
    print(f"f1-score do teste de validação: {f1*100:.2f}%") 

    # matriz de confusão
    cm = confusion_matrix(y_valid, y_pred)

    # calcular percentual por linha p/ cada classe real
    cm_percent = []
    for i, row in enumerate(cm):
        row_total = row.sum() # total de exemplos
        if row_total == 0:
            row_percent = [0]*len(row) # linha de zeros
        else:
            row_percent = (row / row_total) * 100 # percentual por classe
        cm_percent.append(row_percent)
    cm_percent = np.array(cm_percent) 

    print("matriz de confusão %:")
    labels = le.classes_ 
    print(pd.DataFrame(np.round(cm_percent,2), index=labels, columns=labels))


if __name__ == "__main__":
    X_train, y_train, X_valid, y_valid, le = carregar_dados("features_train.csv", "features_valid.csv")
    
    knn(X_train, y_train, X_valid, y_valid)
    svm(X_train, y_train, X_valid, y_valid)
