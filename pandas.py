import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Criando um DataFrame (Dataset Fictício)

# Vamos criar um DataFrame fictício para simular um dataset de exemplo para Machine Learning
data = {
    'idade': [25, 30, 35, 40, 45, 50, 55, 60, np.nan, 28, 33, 38],
    'salario': [50000, 60000, 75000, 90000, 100000, 120000, 130000, 150000, 55000, 65000, np.nan,80000],
    'sexo': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M','F'],
    'cidade': ['São Paulo', 'Rio de Janeiro', 'São Paulo', 'Belo Horizonte', 'Rio de Janeiro', 'São Paulo', 'Belo Horizonte', 'Rio de Janeiro','São Paulo','São Paulo','Rio de Janeiro','Belo Horizonte'],
    'comprou': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0: Não comprou, 1: Comprou
}

df = pd.DataFrame(data)

# 2. Exploração Inicial dos Dados (EDA - Exploratory Data Analysis)

print("Informações Gerais do DataFrame:\n")
print(df.info())  # Informações sobre os tipos de dados e valores não nulos
print("\nEstatísticas Descritivas:\n")
print(df.describe())  # Estatísticas descritivas das colunas numéricas
print("\nPrimeiras 5 Linhas:\n")
print(df.head())  # Exibe as primeiras 5 linhas
print("\nÚltimas 5 Linhas:\n")
print(df.tail()) # Exibe as últimas 5 linhas

# 3. Limpeza de Dados (Data Cleaning)

# Lidar com valores nulos (NaN)
print("\nVerificando valores nulos:\n", df.isnull().sum())
df['idade'] = df['idade'].fillna(df['idade'].mean())  # Preenche os NaN da coluna 'idade' com a média
df['salario'] = df['salario'].fillna(df['salario'].median()) # Preenche os NaN da coluna 'salario' com a mediana
print("\nVerificando valores nulos após tratamento:\n", df.isnull().sum())

# 4. Transformação de Dados (Data Transformation)

# Convertendo a coluna 'sexo' para numérico usando Label Encoding
label_encoder = LabelEncoder()
df['sexo'] = label_encoder.fit_transform(df['sexo'])  # M: 1, F: 0
#Convertendo a coluna "cidade" para numérico usando o One Hot Encoding
df = pd.get_dummies(df, columns=['cidade'])

# 5. Seleção de Features (Feature Selection)

# Definindo as features (X) e o target (y)
X = df.drop('comprou', axis=1)  # Todas as colunas, exceto 'comprou'
y = df['comprou']  # Coluna 'comprou'

# 6. Divisão dos Dados em Treino e Teste (Train/Test Split)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 7. Padronização dos Dados (Feature Scaling)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Treinando um Modelo de Machine Learning (Logistic Regression)

modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# 9. Avaliando o Modelo (Model Evaluation)

y_pred = modelo.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)
print(f"\nAcurácia do modelo: {acuracia:.2f} (x100%)")

# 10. Exibindo o DataFrame Final
print("\nDataFrame final após as transformações:\n",df)
