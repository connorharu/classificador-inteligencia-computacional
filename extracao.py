import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

def carregar_modelo(): # carregamento de resnet pre treinada
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()  # remove a camada de classificação
    model = model.to(device)
    model.eval()
    return model

transform = transforms.Compose([ # diminuir imagens e normalizar
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extrair_features(image_path, model): # extrair features da imagem
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad(): # sem calculo de gradientes pq nao quero treinar agora, e tbm deixa mt pesado
        features = model(img).squeeze().cpu().numpy() # vetor 1d retornado da resnet

    return features

def extrair_pasta(folder_path, model): # extrair features de um diretorio inteiro
    data = []

    for img_name in tqdm(os.listdir(folder_path), desc=f"processando {folder_path}"):
        if img_name.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, img_name)
            try:
                feats = extrair_features(img_path, model) # abre, transforma, passa pelo modelo, pega o vetor e converte pra numpy (basicamente, o fluxo todo para cada imagem)
                data.append([img_path] + feats.tolist()) # adiciona na lista pro df
            except Exception as e:
                print(f"erro - {img_path}: {e}")

    num_features = len(data[0]) - 1 if data else 0 # numero de features a partir da minheira imagem
    columns = ["image_path"] + [f"feat_{i}" for i in range(num_features)] # nome das colunas
    df = pd.DataFrame(data, columns=columns) # cria dataframe
    return df

def save_df(df, path):
    df.to_csv(path, index=False)
    print(f"salvo: {path}")

if __name__ == "__main__":
    model = carregar_modelo()

    base = "simpsons"

    train_folder = os.path.join(base, "train")
    valid_folder = os.path.join(base, "valid")

    # extrair features
    df_train = extrair_pasta(train_folder, model)
    df_valid = extrair_pasta(valid_folder, model)

    save_df(df_train, "features_train.csv")
    save_df(df_valid, "features_valid.csv")
