import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import cv2

# Cargar el conjunto de datos de dígitos
digitos = datasets.load_digits()

# a) Generar una matriz 8x8 con la imagen promedio de cada dígito (0-9)
imagenes_promedio = []
for digito in range(10):
    mascara = digitos.target == digito  # Imágenes que corresponden al dígito actual
    imagen_promedio = digitos.images[mascara].mean(axis=0)  # Promedio de esas imágenes
    imagenes_promedio.append(imagen_promedio)

imagenes_promedio = np.array(imagenes_promedio)

# b) Mostrar las imágenes promedio de manera visual
fig, ejes = plt.subplots(2, 5, figsize=(10, 5))  # 2 filas, 5 columnas
for i, eje in enumerate(ejes.ravel()):  # Iterar sobre todos los subplots
    eje.imshow(imagenes_promedio[i], cmap='gray')  # Mostrar imagen en escala de grises
    eje.set_title(f'Dígito: {i}')
    eje.axis('off')  # Ocultar ejes
plt.show()

# c) Leer un nuevo dígito desde una imagen
def leer_imagen_digito(ruta_imagen):
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)  # Leer en escala de grises
    img_redimensionada = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    img_escalada = (16 * (1 - img_redimensionada / 255)).astype(int)
    return img_escalada

# Reemplaza 'digit_image.png' con la ruta a tu archivo de imagen
nuevo_digito = leer_imagen_digito('imagen5.png')

# d) Encontrar los 3 dígitos más similares al nuevo dígito usando distancia euclidiana
digitos_aplanados = digitos.data  # Aplanar las imágenes a vectores
nuevo_digito_aplanado = nuevo_digito.flatten().reshape(1, -1)

distancias = euclidean_distances(digitos_aplanados, nuevo_digito_aplanado).flatten()
indices_mas_cercanos = np.argsort(distancias)[:3]  # Índices de los 3 más cercanos
digitos_mas_cercanos = digitos.target[indices_mas_cercanos]

# e) Imprimir los valores de los 3 dígitos más cercanos
print(f"Los 3 dígitos más cercanos son: {digitos_mas_cercanos}")

# f) Clasificar el nuevo dígito basado en los 3 más cercanos
if np.all(digitos_mas_cercanos == digitos_mas_cercanos[0]):  # Todos son iguales
    digito_clasificado = digitos_mas_cercanos[0]
else:
    digito_clasificado = np.bincount(digitos_mas_cercanos).argmax()  # El más frecuente

print(f"Soy la inteligencia artificial, y he detectado que el dígito ingresado corresponde al número {digito_clasificado}")

# g) Calcular la distancia a las 10 imágenes promedio e identificar la más cercana
distancias_promedio = euclidean_distances(imagenes_promedio.reshape(10, -1), nuevo_digito_aplanado).flatten()
indice_promedio_mas_cercano = np.argmin(distancias_promedio)
print(f"Soy la inteligencia artificial versión 2, y he detectado que el dígito ingresado corresponde al número {indice_promedio_mas_cercano}")


# Guardar resultados en CSV con 5 grupos para formato condicional
resultados = pd.DataFrame(digitos_aplanados)
resultados['target'] = digitos.target

# Normalizar las distancias al rango [0, 1]
distancias_normalizadas = (distancias - distancias.min()) / (distancias.max() - distancias.min())

# Crear 5 grupos basados en cuantiles
resultados['grupo_distancia'] = pd.qcut(distancias_normalizadas, q=5, labels=False)

resultados['digito_clasificado'] = digito_clasificado
resultados.to_csv("resultados_con_grupos.csv", index=False)

# Mostrar el nuevo dígito para visualización
plt.imshow(nuevo_digito, cmap='gray')
plt.title(f"Nuevo Dígito Clasificado como: {digito_clasificado}")
plt.show()