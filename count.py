import os
import pandas as pd

pasta_atual = os.path.dirname(os.path.abspath(__file__))

arquivos_txt = [f for f in os.listdir(pasta_atual) if f.lower().endswith('.txt')]

print(f"Quantidade de arquivos .txt na pasta: {len(arquivos_txt)}")

arquivos_csv = [f for f in os.listdir(pasta_atual) if f.lower().endswith('.csv')]
print(f"Quantidade de arquivos .csv na pasta: {len(arquivos_csv)}")

csv_path = "solutions.csv"   
txt_folder = "."  

df = pd.read_csv(csv_path, sep=";")

dataset_names = set(df["name"].str.strip())

txt_files = [f for f in os.listdir(txt_folder) if f.endswith(".txt")]

txt_names = set(os.path.splitext(f)[0] for f in txt_files)

missing_in_dataset = txt_names - dataset_names

if missing_in_dataset:
    print("Arquivos TXT que não estão no dataset:")
    for name in sorted(missing_in_dataset):
        print(name)
else:
    print("Todos os arquivos TXT estão no dataset.")

diretorio = "."

txt_files = [f for f in os.listdir(diretorio) if f.endswith(".txt")]
instancias_txt = [os.path.splitext(f)[0] for f in txt_files]

csv_files = [f for f in os.listdir(diretorio) if f.startswith("resultados_") and f.endswith(".csv")]
instancias_csv = [f.replace("resultados_", "").replace(".csv", "") for f in csv_files]

faltando = [inst for inst in instancias_txt if inst not in instancias_csv]

if faltando:
    print("Instâncias ainda não executadas:")
    for inst in faltando:
        print(f"- {inst}")
else:
    print("Todas as instâncias já foram executadas!")