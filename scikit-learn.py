import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris  # Importa um dataset padrão
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carregando um Dataset (Dataset Iris)

# Carregando um dataset padrão do sklearn: o dataset Iris, que é um dataset para classificação.
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# 2. Exploração Inicial dos Dados (EDA)
print("Informações iniciais do dataset:\n",X.info())
print("\nPrimeiras 5 linhas:\n",X.head())
print("\nEstatisticas descritivas:\n",X.describe())

# 3. Divisão dos Dados em Treino e Teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Padronização dos Dados (Feature Scaling)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Treinando um Modelo de Machine Learning (Regressão Logística)

# Criando um modelo
modelo_logistica = LogisticRegression(max_iter=200)

# Treinando o modelo
modelo_logistica.fit(X_train, y_train)

# 6. Avaliando o Modelo (Logistic Regression)
y_pred_logistica = modelo_logistica.predict(X_test)
acuracia_logistica = accuracy_score(y_test, y_pred_logistica)
print(f"\nAcurácia do modelo de Regressão Logística: {acuracia_logistica:.2f}")

#Relatório de Classificação
print("\nRelatório de Classificação do modelo de Regressão Logística:\n",classification_report(y_test,y_pred_logistica))

#Matriz de Confusão
matriz_confusao_logistica = confusion_matrix(y_test,y_pred_logistica)
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao_logistica, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - Regressão Logística')
plt.xlabel('Valores Previstos')
plt.ylabel('Valores Reais')
plt.show()

# 7. Treinando outro Modelo de Machine Learning (SVM)

# Criando um modelo
modelo_svm = SVC()

# Treinando o modelo
modelo_svm.fit(X_train, y_train)

# 8. Avaliando o Modelo (SVM)
y_pred_svm = modelo_svm.predict(X_test)
acuracia_svm = accuracy_score(y_test, y_pred_svm)
print(f"\nAcurácia do modelo SVM: {acuracia_svm:.2f}")

#Relatório de Classificação
print("\nRelatório de Classificação do modelo SVM:\n",classification_report(y_test,y_pred_svm))

#Matriz de Confusão
matriz_confusao_svm = confusion_matrix(y_test,y_pred_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao_svm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - SVM')
plt.xlabel('Valores Previstos')
plt.ylabel('Valores Reais')
plt.show()

# 9. Treinando outro Modelo de Machine Learning (DecisionTree)

# Criando um modelo
modelo_arvore = DecisionTreeClassifier()

# Treinando o modelo
modelo_arvore.fit(X_train, y_train)

# 10. Avaliando o Modelo (DecisionTree)
y_pred_arvore = modelo_arvore.predict(X_test)
acuracia_arvore = accuracy_score(y_test, y_pred_arvore)
print(f"\nAcurácia do modelo Arvore de Decisão: {acuracia_arvore:.2f}")

#Relatório de Classificação
print("\nRelatório de Classificação do modelo Arvore de Decisão:\n",classification_report(y_test,y_pred_arvore))

#Matriz de Confusão
matriz_confusao_arvore = confusion_matrix(y_test,y_pred_arvore)
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao_arvore, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - Arvore de Decisão')
plt.xlabel('Valores Previstos')
plt.ylabel('Valores Reais')
plt.show()

# 11. Treinando outro Modelo de Machine Learning (RandomForest)

# Criando um modelo
modelo_random_forest = RandomForestClassifier()

# Treinando o modelo
modelo_random_forest.fit(X_train, y_train)

# 12. Avaliando o Modelo (RandomForest)
y_pred_random_forest = modelo_random_forest.predict(X_test)
acuracia_random_forest = accuracy_score(y_test, y_pred_random_forest)
print(f"\nAcurácia do modelo Random Forest: {acuracia_random_forest:.2f}")

#Relatório de Classificação
print("\nRelatório de Classificação do modelo Random Forest:\n",classification_report(y_test,y_pred_random_forest))

#Matriz de Confusão
matriz_confusao_random_forest = confusion_matrix(y_test,y_pred_random_forest)
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao_random_forest, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - Random Forest')
plt.xlabel('Valores Previstos')
plt.ylabel('Valores Reais')
plt.show()

# 13. Ajuste de Hiperparâmetros com GridSearchCV

# Definindo os parâmetros a serem testados
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

# Criando o objeto GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

# Treinando o modelo com os parâmetros otimizados
grid.fit(X_train, y_train)

# Melhores parâmetros encontrados
print(f"\nMelhores parâmetros encontrados: {grid.best_params_}")

# Avaliando o modelo com os melhores parâmetros
y_pred_grid = grid.predict(X_test)
acuracia_grid = accuracy_score(y_test, y_pred_grid)
print(f"\nAcurácia do modelo SVM com GridSearchCV: {acuracia_grid:.2f}")

#Relatório de Classificação
print("\nRelatório de Classificação do modelo SVM com GridSearchCV:\n",classification_report(y_test,y_pred_grid))

#Matriz de Confusão
matriz_confusao_grid = confusion_matrix(y_test,y_pred_grid)
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao_grid, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - SVM com GridSearchCV')
plt.xlabel('Valores Previstos')
plt.ylabel('Valores Reais')
plt.show()
