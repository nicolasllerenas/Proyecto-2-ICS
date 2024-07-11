import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Cargar el dataset
digits = load_digits()

# Función para generar la imagen promedio de un dígito
def generate_average_image(digit):
    digit_images = digits.images[digits.target == digit]
    return np.mean(digit_images, axis=0)

# Función para mostrar la imagen promedio
def show_average_image(digit):
    avg_image = generate_average_image(digit)
    plt.imshow(avg_image, cmap='gray')
    plt.title(f"Imagen promedio del dígito {digit}")
    plt.axis('off')
    plt.show()

# Menú principal
def main_menu():
    while True:
        print("\n--- Menú ---")
        print("1. Mostrar imagen promedio de un dígito específico")
        print("2. Mostrar imágenes promedio de todos los dígitos")
        print("3. Salir")
        choice = input("Seleccione una opción: ")

        if choice == '1':
            digit = int(input("Ingrese el dígito (0-9): "))
            if 0 <= digit <= 9:
                show_average_image(digit)
            else:
                print("Dígito inválido. Por favor, ingrese un número entre 0 y 9.")
        elif choice == '2':
            fig, axes = plt.subplots(2, 5, figsize=(12, 6))
            for digit in range(10):
                avg_image = generate_average_image(digit)
                ax = axes[digit // 5, digit % 5]
                ax.imshow(avg_image, cmap='gray')
                ax.set_title(f"Dígito {digit}")
                ax.axis('off')
            plt.tight_layout()
            plt.show()
        elif choice == '3':
            print("¡Hasta luego!")
            break
        else:
            print("Opción inválida. Por favor, intente de nuevo.")

if __name__ == "__main__":
    main_menu()