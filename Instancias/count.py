import glob
import os

def contar_txt(diretorio="."):
    arquivos_txt = glob.glob(os.path.join(diretorio, "*.txt"))
    return len(arquivos_txt)

# Exemplo
print(contar_txt("."))
