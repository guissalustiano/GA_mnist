import pickle
import numpy as np
import random

IMG_SIZE = 28
NUM_GERACOES = 100

# Algoritimos geneticos retirados de https://github.com/GrupoTuringCodes/ws-algoritmos-geneticos

CHANCE_MUT = .5      # Chance de mutação de um peso qualquer
CHANCE_CO = .5      # Chance de crossing over de um peso qualquer
NUM_INDIVIDUOS = 50  # Tamanho da população
NUM_MELHORES = 20     # Número de indivíduos que são mantidos de uma geração para a próxima


def ordenar_lista(lista, ordenacao, decrescente=True):
    return [x for _, x in sorted(zip(ordenacao, lista), key=lambda p: p[0], reverse=decrescente)]


def populacao_aleatoria(n):
    # try with randint
    return [np.random.uniform(-1, 1, (10, IMG_SIZE**2)) for i in range(n)]


def valor_das_acoes(individuo, estado):
    return individuo @ estado


def melhor_jogada(individuo, estado):
    return np.argmax(valor_das_acoes(individuo, estado))


def mutacao(individuo):
    novo_individou = individuo.copy()

    for i in range(len(novo_individou)):
        for j in range(len(novo_individou[i])):
            if random.uniform(0, 1) < CHANCE_MUT:
                novo_individou[i, j] = novo_individou[i, j] * \
                    random.uniform(-2, 2)
            if novo_individou[i, j] > 1:
                novo_individou[i, j] = 1
            if novo_individou[i, j] < -1:
                novo_individou[i, j] = -1

    return novo_individou


def crossover(individuo1, individuo2):
    indivio_filho = individuo1.copy()

    for i in range(len(indivio_filho)):
        for j in range(len(indivio_filho[i])):
            if random.uniform(0, 1) < CHANCE_MUT:
                indivio_filho[i, j] = individuo2[i, j]
    return indivio_filho


def calcular_fitness(superestado, individuo):
    acertos = 0
    for estado, valor_esperado in zip(*superestado):
        palpite = melhor_jogada(individuo, estado)
        if valor_esperado == palpite:
            acertos += 1

    return acertos/len(superestado[0])


def proxima_geracao(populacao, fitness):
    proxima_ger = []

    populacao_mantida = ordenar_lista(populacao, fitness)[0:NUM_MELHORES]

    proxima_ger.extend(populacao_mantida)

    while len(proxima_ger) < NUM_INDIVIDUOS:
        pai_1 = random.choices(populacao_mantida)[0]
        pai_2 = random.choices(populacao_mantida)[0]
        filho = crossover(pai_1, pai_2)
        mutacao(filho)

        proxima_ger.append(filho)
    return proxima_ger


if __name__ == "__main__":
    # dataset from https://www.kaggle.com/pablotab/mnistpklgz
    with open('mnist.pkl', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    # train_x, train_y = train_set

    populacao = populacao_aleatoria(NUM_INDIVIDUOS)

    print('ger | fitness\n----+-' + '-'*5*NUM_INDIVIDUOS)

    for ger in range(NUM_GERACOES):
        fitness = [calcular_fitness(train_set, individuo)
                   for individuo in populacao]
        populacao = proxima_geracao(populacao, fitness)

        print('{:3} |'.format(ger),
              ' '.join('{:1.2f}'.format(s) for s in sorted(fitness[:30], reverse=True)))

        if np.max(fitness) > .85:
            break

    fitness = [calcular_fitness(train_set, individuo)
               for individuo in populacao]
    melhor_individuo = ordenar_lista(populacao, fitness)[0]

    teste_real = calcular_fitness(test_set, melhor_individuo)
    acuracia = teste_real

    print("melhor individuo:", melhor_individuo)
    print("acuracia final:", "{:1.2f}".format(acuracia))
