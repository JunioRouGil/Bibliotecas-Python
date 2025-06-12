import numpy as np

# 1. Criando Arrays (Arrays)

# Criando um array a partir de uma lista
lista = [1, 2, 3, 4, 5]
array1 = np.array(lista)
print("Array criado a partir de uma lista:", array1)

# Criando um array de zeros
array_zeros = np.zeros((3, 4))  # Array 3x4 preenchido com zeros
print("\nArray de zeros:\n", array_zeros)

# Criando um array de uns
array_uns = np.ones((2, 3))  # Array 2x3 preenchido com uns
print("\nArray de uns:\n", array_uns)

# Criando uma matriz identidade
matriz_identidade = np.eye(3) # Matriz identidade 3x3
print("\nMatriz identidade:\n", matriz_identidade)

# Criando um array com valores sequenciais
array_sequencial = np.arange(0, 10, 2)  # De 0 a 10 (exclusive), com passo 2
print("\nArray sequencial:", array_sequencial)

# Criando um array com valores igualmente espaçados
array_linspace = np.linspace(0, 1, 5)  # 5 valores entre 0 e 1 (inclusive)
print("\nArray com valores igualmente espaçados:", array_linspace)

# Criando um array com valores aleatórios
array_aleatorio = np.random.rand(2, 3)  # Array 2x3 com valores aleatórios entre 0 e 1
print("\nArray aleatório:\n", array_aleatorio)

# 2. Atributos de um Array

print("\nAtributos do Array:")
print("Dimensões:", array1.ndim)  # Número de dimensões
print("Formato:", array1.shape)  # Formato (linhas, colunas)
print("Tamanho:", array1.size)  # Número total de elementos
print("Tipo de dado:", array1.dtype)  # Tipo de dado dos elementos

# 3. Operações Aritméticas

array2 = np.array([6, 7, 8, 9, 10])

print("\nOperações Aritméticas:")
print("Soma:", array1 + array2)  # Soma elemento a elemento
print("Subtração:", array2 - array1)  # Subtração elemento a elemento
print("Multiplicação:", array1 * array2)  # Multiplicação elemento a elemento
print("Divisão:", array2 / array1)  # Divisão elemento a elemento
print("Exponenciação:", array1 ** 2) # Eleva cada elemento do array ao quadrado
print("Soma de todos os elementos do array1:", array1.sum())

# 4. Operações com Matrizes

matriz1 = np.array([[1, 2], [3, 4]])
matriz2 = np.array([[5, 6], [7, 8]])

print("\nOperações com Matrizes:")
print("Multiplicação de Matrizes:\n", np.dot(matriz1, matriz2)) # Multiplicação matricial
print("Transposta da matriz1:\n", matriz1.T) # Transposta da matriz
print("Determinante da matriz1:\n",np.linalg.det(matriz1))

# 5. Indexação e Fatiamento (Slicing)

print("\nIndexação e Fatiamento:")
print("Primeiro elemento de array1:", array1[0])
print("Elementos do segundo ao quarto de array1:", array1[1:4])  # Do índice 1 ao 3 (exclusive)
print("Todos os elementos até o terceiro de array1:", array1[:3]) # do inicio até o indice 2
print("Todos os elementos a partir do terceiro de array1:", array1[2:]) # do indice 2 até o final
print("Ultimo elemento do array1:", array1[-1]) # ultimo elemento
matriz3 = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("Elemento na linha 2 coluna 3:\n",matriz3[1,2])
print("Linha 2 da matriz3:\n", matriz3[1,:]) # todos os elementos da segunda linha
print("Coluna 3 da matriz3:\n", matriz3[:,2]) # todos os elementos da terceira coluna

# 6. Funções Universais (ufuncs)

print("\nFunções Universais:")
print("Raiz quadrada:", np.sqrt(array1))  # Raiz quadrada de cada elemento
print("Seno:", np.sin(array1))  # Seno de cada elemento
print("Logaritmo natural:", np.log(array1)) # logaritmo de cada elemento

# 7. Remodelando Arrays (Reshaping)

print("\nRemodelando Arrays:")
array_original = np.arange(12)
print("Array original:", array_original)
array_remodelado = array_original.reshape((3, 4))  # Transforma em uma matriz 3x4
print("Array remodelado:\n", array_remodelado)
array_aplanado = array_remodelado.flatten()
print("Array aplanado:\n", array_aplanado)

# 8. Concatenando Arrays

print("\nConcatenando Arrays:")
array_a = np.array([1, 2, 3])
array_b = np.array([4, 5, 6])
array_concatenado = np.concatenate((array_a, array_b))
print("Arrays concatenados:", array_concatenado)

# 9. Estatística

print("\nEstatística:")
print("Média:", np.mean(array1))
print("Desvio padrão:", np.std(array1))
print("Mediana:", np.median(array1))
print("Valor máximo:", array1.max())
print("Valor mínimo:", array1.min())
