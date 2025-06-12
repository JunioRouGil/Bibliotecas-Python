import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns


# 1. Carregando um Dataset (Dataset Iris)

# Carregando um dataset padrão do sklearn: o dataset Iris, que é um dataset para classificação.
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# 2. Divisão dos Dados em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Treinando um Modelo de Regressão Logística
modelo = LogisticRegression(max_iter=200)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)

# 4. Gráfico de Dispersão (Scatter Plot)
plt.figure(figsize=(8, 6))
plt.scatter(X_test['sepal length (cm)'], X_test['sepal width (cm)'], c=y_test, cmap='viridis')
plt.title('Gráfico de Dispersão: Comprimento x Largura da Sépala')
plt.xlabel('Comprimento da Sépala (cm)')
plt.ylabel('Largura da Sépala (cm)')
plt.colorbar(label='Espécie')
plt.show()

# 5. Histograma
plt.figure(figsize=(8, 6))
plt.hist(X_test['petal length (cm)'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histograma do Comprimento da Pétala')
plt.xlabel('Comprimento da Pétala (cm)')
plt.ylabel('Frequência')
plt.show()

# 6. Gráfico de Linha
# Criando alguns dados fictícios para um exemplo de gráfico de linha
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure(figsize=(8, 6))
plt.plot(x, y, color='green')
plt.title('Gráfico de Linha: Exemplo com Seno')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.show()

# 7. Matriz de Confusão

matriz_confusao = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Valores Previstos')
plt.ylabel('Valores Reais')
plt.show()

# 8. Curva ROC
# Usando apenas para a primeira classe, para simplicidade do exemplo
y_test_bin = np.where(y_test == 0, 1, 0)
y_prob_bin = y_prob[:, 0]

fpr, tpr, thresholds = roc_curve(y_test_bin, y_prob_bin)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# 9. Gráfico de Barras
acuracias = {
    'Regressão Logística': accuracy_score(y_test, y_pred),
}

plt.figure(figsize=(8, 6))
plt.bar(acuracias.keys(), acuracias.values(), color=['skyblue'])
plt.title('Comparação de Acurácias')
plt.xlabel('Modelos')
plt.ylabel('Acurácia')
plt.ylim([0, 1])
plt.show()
