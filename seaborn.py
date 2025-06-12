import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# 1. Carregando o Dataset Iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
df = pd.concat([X, y.rename('species')], axis=1)

# 2. Divisão dos Dados em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Padronização dos Dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Treinamento do Modelo
modelo = LogisticRegression(max_iter=200)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# 5. Análise Exploratória de Dados (EDA) com Seaborn

# a) Gráfico de Dispersão com Hue (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='species', data=df)
plt.title('Gráfico de Dispersão: Comprimento x Largura da Sépala')
plt.show()

# b) Pair Plot
sns.pairplot(df, hue='species', diag_kind='kde')
plt.title("Pair Plot")
plt.show()

# c) Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='petal length (cm)', data=df)
plt.title('Box Plot do Comprimento da Pétala por Espécie')
plt.show()

# d) Violin Plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='species', y='petal width (cm)', data=df)
plt.title('Violin Plot da Largura da Pétala por Espécie')
plt.show()

# e) Heatmap de Correlação
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap da Matriz de Correlação')
plt.show()

# 6. Avaliação de Modelo com Seaborn

# a) Matriz de Confusão
matriz_confusao = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Valores Previstos')
plt.ylabel('Valores Reais')
plt.show()

# b) Relatório de Classificação (não é um gráfico, mas é comum na avaliação)
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
