#CÓDGIDO PARA ESTIMAR CURVA GRANULOMÉTRICA A PARTIR DE IMAGENS 2D DE MICROTOMOGRAFIA

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import pandas as pd

# Diretório das imagens
diretorio_das_imagens = '...\\images\\' #Inserir caminho do diretório com os arquivos das imagens

# Listando todos os arquivos no diretório
arquivos_imagens = os.listdir(diretorio_das_imagens)

# Lista para armazenar os dados
dados = []

# Fator de conversão de micrômetros para milímetros
fator_conversao = 11.32 / 1000  # Inserir resolução obtida na microtomografia (micrômetros por pixel para milímetros)

# Iterando sobre os arquivos e processa cada imagem
for arquivo in arquivos_imagens:
    # Verificando se o arquivo é uma imagem (pode adicionar mais extensões se necessário)
    if arquivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        # Constrói o caminho completo para a imagem
        caminho_completo = os.path.join(diretorio_das_imagens, arquivo)

        # Abrindo a imagem
        imagem = Image.open(caminho_completo)
        
        # Conversão da imagem em "array numpy" para visualização e processamento
        imagem_array = np.array(imagem)
        
        # Inversão das cores
        inv_imagem_array = 255 - imagem_array
            
        # Desfoque gaussiano
        desfoque = cv2.GaussianBlur(inv_imagem_array, (5, 5), 0)    
        
        # Método de Otsu - limiar global, ou seja, em toda a imagem, de forma automática
        valor, otsu = cv2.threshold(desfoque, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        # Plotando a matriz de distância
        dist = ndi.distance_transform_edt(otsu)
        
        # Localizando os máximos locais na matriz de distância
        local_max = peak_local_max(dist,  indices=False, min_distance=20, labels=otsu)
        
        # Rotulando os máximos locais
        markers = ndi.label(local_max, structure=np.ones((3, 3)))[0]
        
        # Aplicando watershed
        labels = watershed(-dist, markers, mask=otsu)
        
        # Individualizando as partículas
        img_final = labels.copy()
        
        for label in np.unique(labels):
            if label == 0:
                continue
            mascara = np.zeros(otsu.shape, dtype='uint8')
            mascara[labels == label] = 255
            cnts = cv2.findContours(mascara.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
            c = max(cnts, key=cv2.contourArea) 
            
            # Calcular momentos da imagem
            momentos = cv2.moments(c)
            
            # Calcular o centróide (x, y) 
            if momentos["m00"] > 0:
                cx = int(momentos["m10"] / momentos["m00"])
                cy = int(momentos["m01"] / momentos["m00"])
            else:
                cx = cy = 0  # Defina cx e cy como 0 se a área for zero
            
            # Calcular tamanho da menor e maior aresta do retângulo
            rotated_rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rotated_rect)
            box = np.int0(box)
            menor_aresta = min(np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[2] - box[1]))
            maior_aresta = max(np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[2] - box[1]))
            
            # Converter tamanho das arestas para milímetros
            menor_aresta_mm = menor_aresta * fator_conversao
            maior_aresta_mm = maior_aresta * fator_conversao
            
            # Calcular área da partícula em pixels
            area_pixel = cv2.contourArea(c)
            
            # Converter área da partícula para milímetros quadrados
            area_mm2 = area_pixel * (fator_conversao ** 2)
            
            # Adicionar dados à lista
            dados.append((arquivo, label, menor_aresta_mm, area_mm2))
        
            # Desenhar o centróide na imagem final
            cv2.circle(img_final, (cx, cy), 5, (0, 0, 0), -1)  # Desenha um círculo no centróide
            
            # Adicionar número da partícula e coordenadas do centróide
            texto = "{} - Centróide: ({}, {})".format(label, cx, cy)
            cv2.putText(img_final, texto, (int(cx) - 1, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 200), 2)
            
            # Desenhar o retângulo delimitador rotacionado em vermelho
            cv2.drawContours(img_final, [box], 0, (255, 0, 0), thickness=2)

# Criar DataFrame com os dados
df = pd.DataFrame(dados, columns=['Nome do Arquivo', 'Número da Partícula', 'Menor Aresta (mm)', 'Área (mm²)'])

# Ordenar o DataFrame com base na coluna 'Menor Aresta (mm)'
df = df.sort_values(by='Menor Aresta (mm)')

# Calcular a soma acumulada da coluna 'Área (mm²)'
df['Área Acumulada (mm²)'] = df['Área (mm²)'].cumsum()

# Calcular o maior valor da coluna 'Área acumulada (mm²)'
max_area_acumulada = df['Área Acumulada (mm²)'].max()

# Calcular a porcentagem da área acumulada
df['Área Acumulada em %'] = (df['Área Acumulada (mm²)'] / max_area_acumulada) * 100

# Ordenar o DataFrame pela coluna 'Menor Aresta (mm)'
df_sorted = df.sort_values(by='Menor Aresta (mm)')

# Plotar o gráfico em escala logarítmica com grades verticais
plt.figure(figsize=(10, 6))
plt.plot(df_sorted['Menor Aresta (mm)'], df_sorted['Área Acumulada em %'], marker='o', linestyle='-')
plt.xscale('log')  # Definir escala logarítmica no eixo x
plt.xlabel('Menor Aresta (mm)')
plt.ylabel('Área Acumulada em %')
plt.title('Gráfico de Menor Aresta vs. Área Acumulada em % em Escala Logarítmica com Grades Verticais')
plt.grid(True, which='both', axis='x', linestyle='--')  # Adicionar grades verticais
plt.grid(True, axis='y')  # Adicionar grades horizontais
plt.show()
