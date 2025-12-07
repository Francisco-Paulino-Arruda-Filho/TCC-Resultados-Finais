#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import random
import pandas as pd
import psutil
import os
import time
import glob
from typing import List, Tuple, Union, Optional
import csv


# In[11]:


def calcular_aptidao_qap(
    solucao: Union[List[int], np.ndarray, List[List[int]]],
    matriz_fluxo: np.ndarray,
    matriz_distancia: np.ndarray
) -> int:
    solucao_flat: List[int] = []
    for x in solucao:
        if isinstance(x, (list, np.ndarray)):
            solucao_flat.extend(int(v) for v in x)
        else:
            solucao_flat.append(int(x))

    n: int = len(solucao_flat)
    custo: int = 0
    for i in range(n):
        for j in range(n):
            custo += matriz_fluxo[i][j] * matriz_distancia[solucao_flat[i]][solucao_flat[j]]
    return custo


# In[12]:


def inicializar_feromonio(n: int, valor_inicial: float = 0.1) -> np.ndarray:
    return np.full((n, n), valor_inicial)


# In[13]:


def construir_solucao_qap(
    feromonio: np.ndarray,
    alpha: float,
    beta: float,
    heuristica: np.ndarray
) -> List[int]:
    """
    feromonio -> memória coletiva das soluções anteriores 
    heuristica -> informação local que guia a escolha 
    alpha -> peso do feromônio
    beta -> peso da heurística
    """
    n: int = feromonio.shape[0]
    nao_visitados: List[int] = list(range(n))
    solucao: List[int] = []

    while nao_visitados:
        if not solucao:
            atual: int = random.choice(nao_visitados)
        else:
            ultima: int = solucao[-1]
            pesos: List[float] = []
            for prox in nao_visitados:
                tau: float = feromonio[ultima][prox] ** alpha
                eta: float = heuristica[ultima][prox] ** beta
                pesos.append(tau * eta)
            pesos_np: np.ndarray = np.array(pesos, dtype=float)
            if not np.isfinite(pesos_np).all() or pesos_np.sum() <= 0:
                pesos_np = np.ones_like(pesos_np) / len(pesos_np)
            else:
                pesos_np /= pesos_np.sum()
            atual = random.choices(nao_visitados, weights=pesos_np)[0]


        solucao.append(atual)
        nao_visitados.remove(atual)

    return solucao


# In[14]:


def atualizar_feromonio(
    feromonio: np.ndarray,
    solucoes: List[List[int]],
    custos: List[int],
    rho: float,
    Q: float
) -> np.ndarray:
    feromonio *= (1 - rho)  
    for solucao, custo in zip(solucoes, custos):
        if custo <= 0 or not np.isfinite(custo):
            custo = 1e-6 
        deposito: float = Q / custo

        for i in range(len(solucao) - 1):
            feromonio[solucao[i]][solucao[i + 1]] += deposito
        feromonio[solucao[-1]][solucao[0]] += deposito
    return feromonio


# In[15]:


def save_aco_iteration_results(
    file_name: str,
    instancia: str,
    ant_index: int,
    best_cost: float,
    current_cost: float,
    tempo_inicio_global: float
) -> None:
    """
    Salva resultados parciais da execução do ACO em um arquivo CSV no mesmo diretório.
    """
    path = file_name  

    header = [
        "instancia",
        "iteracao",
        "formiga",
        "melhor_custo",
        "custo_atual",
        "tempo_decorrido"
    ]
    write_header = not os.path.exists(path)

    with open(path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([
            instancia,
            ant_index,
            best_cost,
            current_cost,
            round(time.time() - tempo_inicio_global, 3)
        ])


# In[16]:


def aco_qap(
    fluxo: np.ndarray,
    distancia: np.ndarray,
    n_formigas: int = 10,
    alpha: float = 1,
    beta: float = 2,
    rho: float = 0.1,
    Q: float = 1,
    instancia: str = "desconhecida",
    tempo_inicio_global: float = None,
    tempo_global_max: float = None
) -> Tuple[List[int], int]:

    n: int = len(fluxo)
    heuristica: np.ndarray = 1 / (np.array(distancia) + 1e-10)
    feromonio: np.ndarray = inicializar_feromonio(n)

    melhor_solucao = None
    melhor_custo = float("inf")

    file_name = f"Resultados/iteracoes_{instancia}_ACO.csv"

    while True:

        if (time.time() - tempo_inicio_global) >= tempo_global_max:
            return melhor_solucao if melhor_solucao else [], int(melhor_custo)

        solucoes = []
        custos = []

        for j in range(n_formigas):

            if (time.time() - tempo_inicio_global) >= tempo_global_max:
                return melhor_solucao if melhor_solucao else [], int(melhor_custo)

            s = construir_solucao_qap(feromonio, alpha, beta, heuristica)
            custo = calcular_aptidao_qap(s, fluxo, distancia)

            solucoes.append(s)
            custos.append(custo)

            if custo < melhor_custo:
                melhor_solucao = s
                melhor_custo = custo

            save_aco_iteration_results(
                file_name=file_name,
                instancia=instancia,
                ant_index=j,
                best_cost=melhor_custo,
                current_cost=custo,
                tempo_inicio_global=tempo_inicio_global
            )

        feromonio = atualizar_feromonio(feromonio, solucoes, custos, rho, Q)

        save_aco_iteration_results(
            file_name=file_name,
            instancia=instancia,
            ant_index=-1,
            best_cost=melhor_custo,
            current_cost=min(custos),
            tempo_inicio_global=tempo_inicio_global
        )


# In[17]:


def ler_qap_com_n(caminho: str) -> Tuple[int, pd.DataFrame, pd.DataFrame]:
    with open(caminho, "r") as f:
        dados: List[int] = list(map(int, f.read().split()))

    n: int = dados[0]
    valores: List[int] = dados[1:]

    total_esperado: int = 2 * n * n
    if len(valores) != total_esperado:
        raise ValueError(f"Esperado {total_esperado} valores, mas encontrado {len(valores)}.")

    flow_flat: List[int] = valores[: n * n]
    dist_flat: List[int] = valores[n * n :]

    flow_df: pd.DataFrame = pd.DataFrame([flow_flat[i * n : (i + 1) * n] for i in range(n)])
    dist_df: pd.DataFrame = pd.DataFrame([dist_flat[i * n : (i + 1) * n] for i in range(n)])

    return n, flow_df, dist_df


# In[ ]:


if __name__ == "__main__":
    arquivos = glob.glob("Instancias/*.txt")

    tempo_global_max = 10 * 60  

    for arquivo in arquivos:
        nome_instancia = os.path.splitext(os.path.basename(arquivo))[0]

        n, flow_df, dist_df = ler_qap_com_n(arquivo)
        matriz_fluxo = np.array(flow_df.values.tolist())
        matriz_distancia = np.array(dist_df.values.tolist())

        resultados = []
        tempo_inicio_instancia = time.time()

        seed = 42
        random.seed(seed)
        np.random.seed(seed)

        rho_value = 0.05 
        process = psutil.Process(os.getpid())

        tempo_inicio_seed = time.time()

        parametros = {
            "n_formigas": 10,
            "alpha": 1.0,
            "beta": 2.0,
            "rho": rho_value,
            "Q": 1,
        }

        melhor_solucao, melhor_custo = aco_qap(
            fluxo=matriz_fluxo,
            distancia=matriz_distancia,
            instancia=nome_instancia,
            tempo_inicio_global=tempo_inicio_instancia,
            tempo_global_max=tempo_global_max,
            **parametros
        )

        tempo_fim_seed = time.time()
        tempo_execucao = tempo_fim_seed - tempo_inicio_instancia

        try:
            memoria_usada = process.memory_info().peak_wset / (1024 * 1024)
        except AttributeError:
            memoria_usada = process.memory_info().rss / (1024 * 1024)

        resultados.append({
            "instancia": nome_instancia,
            "seed": seed,
            "rho": rho_value,
            "melhor_solucao": melhor_solucao,
            "custo": melhor_custo,
            "tempo_execucao_segundos": round(tempo_execucao, 2),
            "memoria_usada_MB": round(memoria_usada, 2),
            **parametros
        })

        print(
            f"[{nome_instancia}] Finalizado em 10 minutos. "
            f"Custo: {melhor_custo} | Memória: {round(memoria_usada, 2)} MB"
        )

        df_resultados = pd.DataFrame(resultados)
        df_resultados.to_csv(f"Resultados/resultados_{nome_instancia}_ACO_10min.csv", index=False)

        print(f"\nResultados da instância {nome_instancia} salvos em resultados_{nome_instancia}_ACO_10min.csv")

