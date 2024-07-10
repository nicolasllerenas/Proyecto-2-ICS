import pandas as pd
from sklearn import datasets

misdatos = datasets.load_digits()
print(misdatos)

print("\nEl value asociado al key target es: ")
print(misdatos["target"])

#Imprime la última de las imagenes planadas
print("\nLa primera planada es:")
print(misdatos["data"][0])
print("n\La última planada es:")
print(misdatos["target"][-1])

#Termine el primer y ultimo target
print("\nLa primera planada es:")
print(misdatos["target"][0])
print("\nLa útimo target es:")
print(misdatos["target"][-1])

#imprime la primera y ultima matriz
print("\nLa primera matriz es:")
print(misdatos["images"][0])

#¿Cuantos digitos hay?
print("\nCantida de elementos:", len(misdatos["data"]))
print("\nCantida de elementos:", len(misdatos["target"]))
print("\nCantida de elementos:", len(misdatos["images"]))

#a partir de ahora, comenzamos a armar el .xls que está subido en Canvas

#primero guardamos la primera matriz en df1
df1 = pd.DataFrame(data=misdatos["images"][0])
print()
print(df1)

#Luego temporalmente guardamos la segunda matriz en df2
df2 = pd.DataFrame(data=misdatos["images"][1])

#Creamos un separador
separador = pd.DataFrame(data=[[0,0,0,0,0,0,0,0]])

#Concatenamos df con el separador
df1 = pd.concat([df1, separador], ignore_index=True)

#Concatenamos df1 y df2
df1 = pd.concat([df1, df2], ignore_index=True)

print("\nVa quedando asi:")
print(df1)

#Ahora, iterativaamente, agregamos el separador y la df2 en el indice "i"
i = 2
while i <= 1796:
    df1 = pd.concat([df1, separador], ignore_index=True)
    df2 = pd.DataFrame(data=misdatos["images"][i])
    df1 = pd.concat([df1, df2], ignore_index=True)
    i = i+1

df1 = pd.concat([df1, separador], ignore_index=True)

#hacer una lista que tenga 9 veces cada target
#los targets están en mis datos["target]

listanueva = []
i = 0
while i <= 1796:
    #hacer esto 9 veces
    j = 0
    while j <=8:
        listanueva.append(misdatos["target"][i])
        j = j +1
    i = i +1

#Agregar esta lista a df1

print()
print(df1)
print(len(listanueva))

df1["nuevaaa"] = listanueva
df1.to_csv("quedadoasi.csv")