#!/usr/bin/env python
# coding: utf-8

# # Imports and types

# In[22]:


import random
import time
import numpy as np
import pandas as pd
import psutil
import os 
import glob
from typing import List, Tuple, Callable
import csv


# In[23]:


def switch_mutation(solucao: List[int]) -> List[int]:
    nova = solucao[:]
    a, b = random.sample(range(len(nova)), 2)
    nova[a], nova[b] = nova[b], nova[a]
    return nova


# In[24]:


def permutation_mutation(solucao: List[int]) -> List[int]:
    nova = solucao[:]
    a, b = sorted(random.sample(range(len(nova)), 2))
    nova[a:b+1] = reversed(nova[a:b+1])
    return nova


# In[25]:

ultimo_tempo_decorrido_s = None
def save_es_vnd_iteration_results(file_name: str,
                                  instancia: str,
                                  iteration: int,
                                  melhor_aptidao: float,
                                  melhor_solucao: List[int],
                                  tempo_decorrido: float,
                                  memoria_usada: float,
                                  parametros: dict):
    """
    Salva os resultados parciais de uma execução do ES_VND em um arquivo CSV.
    Cria o arquivo com cabeçalho na primeira execução e adiciona novas linhas a cada iteração.

    Parâmetros
    ----------
    file_name : str
        Nome do arquivo CSV a ser salvo.
    instancia : str
        Nome da instância sendo executada.
    iteration : int
        Iteração atual da ES.
    melhor_aptidao : float
        Valor da melhor aptidão encontrada até o momento.
    melhor_solucao : List[int]
        Solução correspondente à melhor aptidão.
    tempo_decorrido : float
        Tempo decorrido desde o início da execução (em segundos).
    memoria_usada : float
        Memória utilizada (em MB).
    parametros : dict
        Dicionário com parâmetros da execução (mu, lambda, taxas, n, seed etc.).
    """

    if(ultimo_tempo_decorrido_s != None and abs(ultimo_tempo_decorrido_s - tempo_decorrido) < 2:
        return
    ultimo_tempo_decorrido_s = tempo_decorrido

    file_exists = os.path.isfile(file_name)

    with open(file_name, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "instancia",
                "iteration",
                "melhor_custo",
                "tempo_decorrido_s",
                "memoria_usada_MB",
                "mu",
                "lambd",
                "taxa_mutacao",
                "taxa_busca_local",
                "iter_sem_melhora_max",
                "n",
                "seed",
                "melhor_solucao"
            ])

        writer.writerow([
            instancia,
            iteration,
            -melhor_aptidao,  
            round(tempo_decorrido, 2),
            round(memoria_usada, 2),
            parametros.get("mu"),
            parametros.get("lambd"),
            round(parametros.get("taxa_mutacao", 0), 4),
            round(parametros.get("taxa_busca_local", 0), 4),
            parametros.get("iter_sem_melhora_max"),
            parametros.get("n"),
            parametros.get("seed"),
            melhor_solucao
        ])


# In[26]:


# M1: Troca de elementos adjacentes
def gerar_vizinhanca_1(solucao: List[int]) -> List[List[int]]:
    vizinhanca = []
    for i in range(len(solucao) - 1):
        vizinho = solucao[:]
        vizinho[i], vizinho[i+1] = vizinho[i+1], vizinho[i]
        vizinhanca.append(vizinho)
    return vizinhanca


# In[27]:


# M2: Troca de elementos com distância 2
def gerar_vizinhanca_2(solucao: List[int]) -> List[List[int]]:
    vizinhanca = []
    for i in range(len(solucao) - 2):
        vizinho = solucao[:]
        vizinho[i], vizinho[i+2] = vizinho[i+2], vizinho[i]
        vizinhanca.append(vizinho)
    return vizinhanca


# In[28]:


# M3: Todas as trocas possíveis entre pares de posições
def gerar_vizinhanca_3(solucao: List[int]) -> List[List[int]]:
    vizinhanca = []
    n = len(solucao)
    for i in range(n):
        for j in range(i + 1, n):
            vizinho = solucao[:]
            vizinho[i], vizinho[j] = vizinho[j], vizinho[i]
            vizinhanca.append(vizinho)
    return vizinhanca


# In[29]:


# M4: Inserção de um elemento em posição anterior
def gerar_vizinhanca_4(solucao: List[int]) -> List[List[int]]:
    vizinhanca = []
    n = len(solucao)
    for i in range(n):
        for j in range(i):
            vizinho = solucao[:]
            elem = vizinho.pop(i)
            vizinho.insert(j, elem)
            vizinhanca.append(vizinho)
    return vizinhanca


# In[30]:


# M5: Inserção de um elemento em posição posterior
def gerar_vizinhanca_5(solucao: List[int]) -> List[List[int]]:
    vizinhanca = []
    n = len(solucao)
    for i in range(n):
        for j in range(i+1, n):
            vizinho = solucao[:]
            elem = vizinho.pop(i)
            vizinho.insert(j, elem)
            vizinhanca.append(vizinho)
    return vizinhanca


# In[31]:


def gerar_vizinhanca_6(solucao: List[int]) -> List[List[int]]:
    vizinhanca = []
    n = len(solucao)
    # percorre pares consecutivos (u,x) e (v,y)
    for i in range(n-1):
        for j in range(i+2, n-1):
            vizinho = solucao[:]
            vizinho[i], vizinho[i+1], vizinho[j], vizinho[j+1] = (
                vizinho[j], vizinho[j+1], vizinho[i], vizinho[i+1]
            )
            vizinhanca.append(vizinho)
    return vizinhanca


# In[32]:


def gerar_vizinhanca_7(solucao: List[int]) -> List[List[int]]:
    vizinhanca = []
    n = len(solucao)
    for i in range(n-1):
        for j in range(i+2, n-1):  
            vizinho = solucao[:]
            u, x, v, y = vizinho[i], vizinho[i+1], vizinho[j], vizinho[j+1]
            # substitui (u,x) e (v,y) por (u,v) e (x,y)
            vizinho[i], vizinho[i+1], vizinho[j], vizinho[j+1] = u, v, x, y
            vizinhanca.append(vizinho)
    return vizinhanca


# In[33]:


# M8: 2-opt (reversão de qualquer sublista)
def gerar_vizinhanca_8(solucao: List[int]) -> List[List[int]]:
    vizinhanca = []
    n = len(solucao)
    for i in range(n):
        for j in range(i+2, n):  
            vizinho = solucao[:]
            vizinho[i:j] = reversed(vizinho[i:j])
            vizinhanca.append(vizinho)
    return vizinhanca


# In[34]:


# M9: Swap de blocos de tamanho 2
def gerar_vizinhanca_9(solucao: List[int]) -> List[List[int]]:
    vizinhanca = []
    n = len(solucao)
    for i in range(n-1):
        for j in range(i+2, n-1):
            vizinho = solucao[:]
            vizinho[i:i+2], vizinho[j:j+2] = vizinho[j:j+2], vizinho[i:i+2]
            vizinhanca.append(vizinho)
    return vizinhanca


# In[35]:


def gerar_array_replicavel(seed: int, tamanho: int) -> list[int]:
    random.seed(seed)
    vetor = list(range(tamanho))  
    random.shuffle(vetor)
    return vetor


# In[36]:


def calcular_aptidao_qap(solucao: List[int],
                         matriz_fluxo: pd.DataFrame,
                         matriz_distancia: pd.DataFrame) -> int:
    n = len(solucao)
    custo = 0
    for i in range(n):
        for j in range(n):
            custo += matriz_fluxo[i][j] * matriz_distancia[solucao[i]][solucao[j]]
    return -custo   # agora retorna custo negativo (para minimização)


# In[37]:


def calcular_melhor_vizinho(vizinhanca: List[List[int]],
                            matriz_fluxo: pd.DataFrame,
                            matriz_distancia: pd.DataFrame) -> List[int]:
    melhor_vizinho = vizinhanca[0]
    melhor_aptidao = calcular_aptidao_qap(vizinhanca[0], matriz_fluxo, matriz_distancia)
    for i in range(1, len(vizinhanca)):
        vizinho_aptidao = calcular_aptidao_qap(vizinhanca[i], matriz_fluxo, matriz_distancia)
        if  vizinho_aptidao > melhor_aptidao:
            melhor_vizinho = vizinhanca[i]
            melhor_aptidao = vizinho_aptidao
    return melhor_vizinho


# In[38]:


def VND(solucao: List[int],
        matriz_fluxo: pd.DataFrame,
        matriz_distancia: pd.DataFrame,
        tempo_inicio_global: float = None,
        tempo_global_max: float = None
        ) -> List[int]:
    solucao_atual = solucao[:]
    k = 0
    funcoes_vizinhanca: List[Callable[[List[int]], List[List[int]]]] = [
        gerar_vizinhanca_1, gerar_vizinhanca_2, gerar_vizinhanca_3,
        gerar_vizinhanca_4, gerar_vizinhanca_5, gerar_vizinhanca_6,
        gerar_vizinhanca_7, gerar_vizinhanca_8, gerar_vizinhanca_9,
    ]

    while k < len(funcoes_vizinhanca):

        if tempo_inicio_global and tempo_global_max and (time.time() - tempo_inicio_global) > tempo_global_max:
            return solucao_atual
        
        vizinhos = funcoes_vizinhanca[k](solucao_atual)
        aptidao_atual = calcular_aptidao_qap(solucao_atual, matriz_fluxo, matriz_distancia)

        melhorou = False
        for vizinho in vizinhos:
            if tempo_inicio_global and tempo_global_max and (time.time() - tempo_inicio_global) > tempo_global_max:
                return solucao_atual
            
            aptidao_vizinho = calcular_aptidao_qap(vizinho, matriz_fluxo, matriz_distancia)
            if aptidao_vizinho > aptidao_atual:
                solucao_atual = vizinho
                k = 0
                melhorou = True
                break
        
        if not melhorou:
            k += 1

    return solucao_atual


# In[39]:


def inicializar_populacao(lambd: int, tamanho: int, seed_base: int = 42) -> List[List[int]]:
    populacao = []
    for i in range(lambd):
        seed = seed_base + i  
        individuo = gerar_array_replicavel(seed=seed, tamanho=tamanho)
        populacao.append(individuo)
    return populacao


# In[40]:


def ES_VND(instancia: str,
           mu: int,
           lambd: int,
           taxa_mutacao: float,
           taxa_busca_local: float,
           n: int,
           matriz_fluxo: pd.DataFrame,
           matriz_distancia: pd.DataFrame,
           seed: int = 42,
           solucao_inicial: List[int] = None,
           tempo_inicio_global: float = None,
           tempo_global_max: float = None,
           parametros: dict = None
           ) -> Tuple[List[int], float]:
    
    if parametros is None:
        parametros = {}

    P = inicializar_populacao(mu + lambd, tamanho=n, seed_base=seed)

    melhor: List[int] | None = None
    melhor_aptidao = -float('inf')
    iteration = 0
    inicio = time.time()

    while True:
        iteration += 1

        if tempo_inicio_global and tempo_global_max:
            if (time.time() - tempo_inicio_global) > tempo_global_max:
                return (
                    melhor if melhor is not None else solucao_inicial,
                    melhor_aptidao if melhor is not None else calcular_aptidao_qap(
                        solucao_inicial, matriz_fluxo, matriz_distancia
                    )
                )

        aptidoes = [calcular_aptidao_qap(ind, matriz_fluxo, matriz_distancia) for ind in P]

        for ind, apt in zip(P, aptidoes):
            if apt > melhor_aptidao:
                melhor_aptidao = apt
                melhor = ind[:]

        process = psutil.Process(os.getpid())
        try:
            memoria_usada = process.memory_info().peak_wset / (1024 * 1024)
        except AttributeError:
            memoria_usada = process.memory_info().rss / (1024 * 1024)

        tempo_decorrido = time.time() - inicio

        save_es_vnd_iteration_results(
            file_name=f"Resultados/iteracoes_{instancia}_ES_VND.csv",
            instancia=instancia,
            iteration=iteration,
            melhor_aptidao=melhor_aptidao,
            melhor_solucao=melhor,
            tempo_decorrido=tempo_decorrido,
            memoria_usada=memoria_usada,
            parametros=parametros
        )

        melhores_indices = sorted(range(len(P)), key=lambda i: aptidoes[i], reverse=True)[:mu]
        Q = [P[i][:] for i in melhores_indices]

        nova_geracao: List[List[int]] = Q[:]

        for q in Q:

            if tempo_inicio_global and tempo_global_max:
                if (time.time() - tempo_inicio_global) > tempo_global_max:
                    return (
                        melhor if melhor is not None else solucao_inicial,
                        melhor_aptidao if melhor is not None else calcular_aptidao_qap(
                            solucao_inicial, matriz_fluxo, matriz_distancia
                        )
                    )

            for _ in range(lambd // mu):
                individuo = q[:]

                if random.random() < taxa_mutacao:
                    individuo = permutation_mutation(individuo)

                if random.random() < taxa_busca_local:
                    individuo = VND(
                        individuo,
                        matriz_fluxo,
                        matriz_distancia,
                        tempo_inicio_global,
                        tempo_global_max
                    )

                nova_geracao.append(individuo)

                save_es_vnd_iteration_results(
                    file_name=f"Resultados/iteracoes_{instancia}_ES_VND.csv",
                    instancia=instancia,
                    iteration=iteration,
                    melhor_aptidao=melhor_aptidao,
                    melhor_solucao=melhor,
                    tempo_decorrido=time.time() - inicio,
                    memoria_usada=memoria_usada,
                    parametros=parametros
                )

        P = nova_geracao


# In[41]:


def ler_qap_com_n(caminho: str) -> Tuple[int, pd.DataFrame, pd.DataFrame]:
    with open(caminho, "r") as f:
        dados = list(map(int, f.read().split()))

    n = dados[0]
    valores = dados[1:]

    total_esperado = 2 * n * n
    if len(valores) != total_esperado:
        raise ValueError(f"Esperado {total_esperado} valores, mas encontrado {len(valores)}.")

    flow_flat = valores[:n * n]
    dist_flat = valores[n * n:]

    flow_df = pd.DataFrame([flow_flat[i * n:(i + 1) * n] for i in range(n)])
    dist_df = pd.DataFrame([dist_flat[i * n:(i + 1) * n] for i in range(n)])

    return n, flow_df, dist_df


# In[ ]:


if __name__ == "__main__":
    arquivos = glob.glob("Instancias/*.txt")

    # Limite global de tempo por instância (10 minutos)
    tempo_global_max = 10 * 60  

    for arquivo in arquivos:
        nome_instancia = os.path.splitext(os.path.basename(arquivo))[0]
        print(f"Processando instância {nome_instancia} (arquivo: {arquivo})")

        n, flow_df, dist_df = ler_qap_com_n(arquivo)
        matriz_fluxo = np.array(flow_df.values.tolist())
        matriz_distancia = np.array(dist_df.values.tolist())

        resultados = []

        seed = 42
        random.seed(seed)

        process = psutil.Process(os.getpid())
        tempo_inicio_instancia = time.time()
        tempo_inicio = time.time()

        solucao_inicial = gerar_array_replicavel(seed=seed, tamanho=n)

        parametros = {
            "instancia": nome_instancia,
            "mu": 5,
            "lambd": 30,
            "taxa_mutacao": 0.40,
            "taxa_busca_local": 0.40,
            "n": n,
            "seed": seed,
        }

        melhor_solucao, melhor_valor = ES_VND(
            instancia=nome_instancia,
            mu=parametros["mu"],
            lambd=parametros["lambd"],
            taxa_mutacao=parametros["taxa_mutacao"],
            taxa_busca_local=parametros["taxa_busca_local"],
            n=parametros["n"],
            matriz_fluxo=matriz_fluxo,
            matriz_distancia=matriz_distancia,
            seed=parametros["seed"],
            solucao_inicial=solucao_inicial,
            tempo_inicio_global=tempo_inicio_instancia,
            tempo_global_max=tempo_global_max,
            parametros=parametros
        )

        tempo_fim = time.time()
        tempo_decorrido = min(
            tempo_fim - tempo_inicio,
            tempo_global_max
        )

        try:
            memoria_usada = process.memory_info().peak_wset / (1024 * 1024)
        except AttributeError:
            memoria_usada = process.memory_info().rss / (1024 * 1024)

        resultados.append({
            "instancia": nome_instancia,
            "seed": seed,
            "melhor_solucao": melhor_solucao,
            "custo": -melhor_valor,
            "tempo_execucao_segundos": round(tempo_decorrido, 2),
            "memoria_usada_MB": round(memoria_usada, 2)
        })

        print(f"[{nome_instancia}] Finalizada com seed 42. Custo: {(-melhor_valor)} | Memória usada: {round(memoria_usada, 2)} MB")

        df_resultados = pd.DataFrame(resultados)
        df_resultados.to_csv(f"Resultados/resultados_{nome_instancia}_ES_VND_10min.csv", index=False)

        print(f"\nResultados da instância {nome_instancia} salvos em resultados_{nome_instancia}_ES_VND_10min.csv")

