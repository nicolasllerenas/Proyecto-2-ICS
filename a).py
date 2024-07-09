import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

#Cargamos el dataset
digits = load_digits()
#Inicializar un array para guardar las imágenes promedio
average_images =np.zeros((10, 8, 8))