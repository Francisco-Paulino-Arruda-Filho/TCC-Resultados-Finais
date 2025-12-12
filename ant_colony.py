#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import pandas as pd
import psutil
import os
import time
import glob
from typing import List, Tuple, Union
import csv

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


def inicializar_feromonio(n: int, valor_inicial: float = 0.1) -> np.ndarray:
    return np.full((n, n), valor_inicial)


def construir_solucao_qap(
    feromonio: np.ndarray,
    alpha: float,
    beta: float,
    heuristica: np.ndarray
) -> List[int]:
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

def save_aco_iteration_results(
    file_name: str,
    instancia: str,
    formiga: int,
    melhor_custo: float,       
    tempo_decorrido_s: float,
    memoria_usada_MB: float,
    permutacao: List[int]
) -> None:
    pasta = os.path.dirname(file_name)
    if pasta and not os.path.exists(pasta):
        os.makedirs(pasta, exist_ok=True)

    file_exists = os.path.exists(file_name)

    with open(file_name, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "instancia",
                "formiga",
                "melhor_custo",         
                "tempo_decorrido_s",
                "memoria_usada_MB",
                "permutacao"
            ])
        writer.writerow([
            instancia,
            formiga,
            melhor_custo,                
            round(tempo_decorrido_s, 3),
            round(memoria_usada_MB, 3),
            str(permutacao)
        ])

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

    if tempo_inicio_global is None:
        tempo_inicio_global = time.time()

    n: int = len(fluxo)
    heuristica: np.ndarray = 1 / (np.array(distancia) + 1e-10)
    feromonio: np.ndarray = inicializar_feromonio(n)

    melhor_solucao = None
    melhor_custo = float("inf")

    file_name = f"Resultados/iteracoes_{instancia}_ACO.csv"

    process = psutil.Process(os.getpid())

    while True:
        if tempo_global_max is not None and (time.time() - tempo_inicio_global) >= tempo_global_max:
            best_cost_int = int(melhor_custo) if np.isfinite(melhor_custo) else -1
            return melhor_solucao if melhor_solucao else [], best_cost_int

        solucoes = []
        custos = []

        for j in range(n_formigas):

            if tempo_global_max is not None and (time.time() - tempo_inicio_global) >= tempo_global_max:
                best_cost_int = int(melhor_custo) if np.isfinite(melhor_custo) else -1
                return melhor_solucao if melhor_solucao else [], best_cost_int

            s = construir_solucao_qap(feromonio, alpha, beta, heuristica)
            custo = calcular_aptidao_qap(s, fluxo, distancia)

            solucoes.append(s)
            custos.append(custo)

            if custo < melhor_custo:
                melhor_solucao = s
                melhor_custo = custo

            tempo_decorrido = time.time() - tempo_inicio_global
            try:
                memoria_usada = process.memory_info().peak_wset / (1024 * 1024)
            except AttributeError:
                memoria_usada = process.memory_info().rss / (1024 * 1024)

            save_aco_iteration_results(
                file_name=file_name,
                instancia=instancia,
                formiga=j,
                melhor_custo=custo,     
                tempo_decorrido_s=tempo_decorrido,
                memoria_usada_MB=memoria_usada,
                permutacao=s
            )

        feromonio = atualizar_feromonio(feromonio, solucoes, custos, rho, Q)

        tempo_decorrido = time.time() - tempo_inicio_global
        try:
            memoria_usada = process.memory_info().peak_wset / (1024 * 1024)
        except AttributeError:
            memoria_usada = process.memory_info().rss / (1024 * 1024)

        save_aco_iteration_results(
            file_name=file_name,
            instancia=instancia,
            formiga=-1,
            melhor_custo=min(custos), 
            tempo_decorrido_s=tempo_decorrido,
            memoria_usada_MB=memoria_usada,
            permutacao=melhor_solucao
        )

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

if __name__ == "__main__":
    arquivos = glob.glob("Instancias/*.txt")
    tempo_global_max = 10 * 60  # 10 min

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

        parametros = {
            "n_formigas": 10,
            "alpha": 1.0,
            "beta": 2.0,
            "rho": rho_value,
            "Q": 1,
        }

        if not os.path.exists("Resultados"):
            os.makedirs("Resultados", exist_ok=True)

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

        process = psutil.Process(os.getpid())
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
            f"[{nome_instancia}] Finalizado. "
            f"Custo: {melhor_custo} | Memória: {round(memoria_usada, 2)} MB"
        )

        df_resultados = pd.DataFrame(resultados)
        df_resultados.to_csv(f"Resultados/resultados_{nome_instancia}_ACO_10min.csv", index=False)

        print(f"\nResultados da instância {nome_instancia} salvos em Resultados/resultados_{nome_instancia}_ACO_10min.csv")
