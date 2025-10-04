import numpy as np

def one_hot_encode(sentenca):
    palavras = sentenca.lower().split()
    vocabulario = sorted(set(palavras))
    palavra_para_indice = {palavra: i for i,
                           palavra in enumerate(vocabulario)}
    matrix_one_hot = np.zeros((
        len(palavras), len(vocabulario)), dtype=int)
    for i, palavra in enumerate(palavras):
        matrix_one_hot[i, palavra_para_indice[palavra]] = 1
    return matrix_one_hot, vocabulario

sentenca = "Eu amo aprender Python e Python é incrível, apesar de exigir muito da minha cognição."    
matrix_one_hot, vocabulario = one_hot_encode(sentenca)
print("Matriz One-Hot:\n", matrix_one_hot)
print("Vocabulário:\n", vocabulario)
