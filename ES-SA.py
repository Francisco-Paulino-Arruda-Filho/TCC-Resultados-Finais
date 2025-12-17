#!/usr/bin/env python
# coding: utf-8

# # Imports and types

# In[34]:


import random
import time
import numpy as np
import pandas as pd
import psutil
import os 
import glob
from typing import List, Tuple
import csv


# In[35]:


def switch_mutation(solucao: List[int]) -> List[int]:
    nova = solucao[:]
    a, b = random.sample(range(len(nova)), 2)
    nova[a], nova[b] = nova[b], nova[a]
    return nova


# In[36]:


def permutation_mutation(solucao: List[int]) -> List[int]:
    nova = solucao[:]
    a, b = sorted(random.sample(range(len(nova)), 2))
    nova[a:b+1] = reversed(nova[a:b+1])
    return nova


# In[37]:


def gerar_array_replicavel(seed: int, tamanho: int) -> list[int]:
    random.seed(seed)
    vetor = list(range(tamanho))  
    random.shuffle(vetor)
    return vetor


# In[38]:


def calcular_aptidao_qap(solucao: List[int],
                         matriz_fluxo: pd.DataFrame,
                         matriz_distancia: pd.DataFrame) -> int:
    n = len(solucao)
    custo = 0
    for i in range(n):
        for j in range(n):
            custo += matriz_fluxo[i][j] * matriz_distancia[solucao[i]][solucao[j]]
    return -custo


# In[39]:


def recozimento_simulado(solucao_inicial: List[int],
                        matriz_fluxo: pd.DataFrame,
                        matriz_distancia: pd.DataFrame,
                        temperatura_inicial: float,
                        taxa_resfriamento: float,
                        iteracoes_por_temperatura: int,
                        tempo_inicio_global: float = None,
                        tempo_global_max: float = None,
                        instancia: str = None,
                        seed: int = None,
                        mu: int = None,
                        lambd_: int = None,
                        taxa_mutacao: float = None,
                        taxa_busca_local: float = None,
                        iter_sem_melhora_max: int = None,
                        n: int = None,
                        save_path: str = "resultados_parciais.csv"
                        ) -> List[int]:
    """
    Recozimento Simulado com salvamento periódico de progresso.
    """
    solucao_atual = solucao_inicial[:]
    aptidao_atual = calcular_aptidao_qap(solucao_atual, matriz_fluxo, matriz_distancia)
    melhor_solucao = solucao_atual[:]
    melhor_aptidao = aptidao_atual

    temperatura = temperatura_inicial
    process = psutil.Process(os.getpid())

    iteration = 0
    while temperatura > 1:
        for _ in range(iteracoes_por_temperatura):
            iteration += 1
            vizinho = switch_mutation(solucao_atual)
            aptidao_vizinho = calcular_aptidao_qap(vizinho, matriz_fluxo, matriz_distancia)

            delta_aptidao = aptidao_vizinho - aptidao_atual

            if delta_aptidao > 0 or random.uniform(0, 1) < np.exp(delta_aptidao / temperatura):
                solucao_atual = vizinho
                aptidao_atual = aptidao_vizinho

                if aptidao_atual > melhor_aptidao:
                    melhor_solucao = solucao_atual[:]
                    melhor_aptidao = aptidao_atual

            if iteration % 100 == 0:  
                tempo_decorrido = time.time() - tempo_inicio_global
                memoria_usada = process.memory_info().rss / (1024 * 1024)
                with open(save_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        instancia,
                        iteration,
                        -melhor_aptidao,  
                        round(tempo_decorrido, 2),
                        round(memoria_usada, 2),
                        mu,
                        lambd_,
                        round(taxa_mutacao, 4),
                        round(taxa_busca_local, 4),
                        iter_sem_melhora_max,
                        n,
                        seed,
                        melhor_solucao
                    ])
                    f.flush()

            if tempo_inicio_global and tempo_global_max and (time.time() - tempo_inicio_global) >= tempo_global_max:
                    return melhor_solucao

        temperatura *= taxa_resfriamento

    return melhor_solucao


# In[40]:


def inicializar_populacao(lambd: int, tamanho: int, seed_base: int = 42) -> List[List[int]]:
    populacao = []
    for i in range(lambd):
        seed = seed_base + i  
        individuo = gerar_array_replicavel(seed=seed, tamanho=tamanho)
        populacao.append(individuo)
    return populacao


# In[41]:

ultimo_tempo_decorrido_s = None
def save_es_sa_iteration_results(file_name: str,
                                 instancia: str,
                                 iteration: int,
                                 melhor_aptidao: float,
                                 melhor_solucao: List[int],
                                 tempo_decorrido: float,
                                 memoria_usada: float,
                                 parametros: dict):
    """
    Salva os resultados parciais de uma execução do ES_SA em um arquivo CSV.
    Cria o arquivo com cabeçalho na primeira execução e adiciona novas linhas a cada iteração.
    """
    if ultimo_tempo_decorrido_s != None and abs(ultimo_tempo_decorrido_s - tempo_decorrido) < 2:
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
            -melhor_aptidao,  # custo positivo
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


# In[42]:


def ES_SA(mu: int, lambd: int,
          taxa_mutacao: float, taxa_busca_local: float,
          n: int,
          matriz_fluxo: pd.DataFrame, matriz_distancia: pd.DataFrame,
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
    inicio = time.time()
    iteration = 0

    while True:
        iteration += 1

        if tempo_inicio_global and tempo_global_max:
            if (time.time() - tempo_inicio_global) >= tempo_global_max:
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

        save_es_sa_iteration_results(
            file_name=f"Resultados/iteracoes_{parametros.get('instancia', 'desconhecida')}_ES_SA.csv",
            instancia=parametros.get('instancia', 'desconhecida'),
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
                if (time.time() - tempo_inicio_global) >= tempo_global_max:
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
                    individuo = recozimento_simulado(
                        solucao_inicial=individuo,
                        matriz_fluxo=matriz_fluxo,
                        matriz_distancia=matriz_distancia,
                        temperatura_inicial=1000.0,
                        taxa_resfriamento=0.90,
                        iteracoes_por_temperatura=100,
                        tempo_inicio_global=tempo_inicio_global,
                        tempo_global_max=tempo_global_max,
                        instancia=parametros.get("instancia", "desconhecida"),
                        seed=seed,
                        save_path=f"Resultados/iteracoes_{parametros.get('instancia','desconhecida')}_ES_SA.csv",
                        mu=mu,
                        lambd_=lambd,
                        taxa_mutacao=taxa_mutacao,
                        taxa_busca_local=taxa_busca_local,
                        n=n
                    )

                nova_geracao.append(individuo)

        P = nova_geracao


# In[43]:


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
    tempo_global_max = 10 * 60  # 10 minutos por instância

    for arquivo in arquivos:
        nome_instancia = os.path.splitext(os.path.basename(arquivo))[0]
        n, flow_df, dist_df = ler_qap_com_n(arquivo)
        matriz_fluxo = np.array(flow_df.values.tolist())
        matriz_distancia = np.array(dist_df.values.tolist())

        resultados = []
        tempo_inicio_instancia = time.time()

        # AGORA: Apenas 1 seed
        seed = 42

        random.seed(seed)
        process = psutil.Process(os.getpid())
        tempo_inicio = time.time()

        solucao_inicial = gerar_array_replicavel(seed=seed, tamanho=n)

        parametros = {
            "instancia": nome_instancia,
            "mu": 5,
            "lambd": 30,
            "taxa_mutacao": 0.40,
            "taxa_busca_local": 0.40,
            "matriz_fluxo": matriz_fluxo,
            "matriz_distancia": matriz_distancia,
            "solucao_inicial": solucao_inicial,
            "n": n,
            "seed": seed
        }

        parametros_execucao = {k: v for k, v in parametros.items() if k != "instancia"}

        melhor_solucao, melhor_valor = ES_SA(
            **parametros_execucao,
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
            "mu": parametros["mu"],
            "lambd": parametros["lambd"],
            "taxa_mutacao": round(parametros["taxa_mutacao"], 3),
            "taxa_busca_local": round(parametros["taxa_busca_local"], 3),
            "custo": -melhor_valor,
            "tempo_execucao_segundos": round(tempo_decorrido, 2),
            "memoria_usada_MB": round(memoria_usada, 2),
            "melhor_solucao": melhor_solucao,
        })

        print(f"[{nome_instancia}] Finalizada com seed 42. "
              f"Custo: {(-melhor_valor)} | Memória usada: {round(memoria_usada, 2)} MB")

        df_resultados = pd.DataFrame(resultados)
        df_resultados.to_csv(f"Resultados/resultados_{nome_instancia}_ES_SA_10min.csv", index=False)

        print(f"\n Resultados da instância {nome_instancia} salvos em "
              f"Resultados/resultados_{nome_instancia}_ES_SA_10min.csv")

