import numpy as np
# from matplotlib import pyplot as plt

# a. Erzeugen Sie einen Vektor
n = ""
liste = []
## prüfen die Variable n ob es ein Zahl ist
while not n.isdigit():
    n = (input("Geben Sie die Größe des Vektors: "))
n = int(n)

for i in range(n):
    zahl = i * 1 / n
    liste.append(zahl)
x = np.array(liste)
#print(x)

# b. Berechnen Sie den Vektor
new_liste = []
for el in liste:
    new_val = np.sin(np.pi*el)
    new_liste.append(new_val)
y = np.array(new_liste)
#print(new_vek)

#c. Erstellen ein Nx3 Matrix:
liste_mat = []
for i in range(n):
    liste1 = [1, liste[i], liste[i]**2]
    liste_mat.append(liste1)
A = np.array(liste_mat)
#print(mat_N_3)

#d. Erstellung die transpose Matrix:
M = A.T @ A# (3, 3)
#print(M.shape)
#print(M)
b = A.T @ y# (3,1)
#print(b.shape)
#print(b)


# e. Das lineare Gleichungssystem : Ma = b
a = b @ np.linalg.inv(M)# (3,1)
#print(a)

#f.  berechnen die Vektor z:
liste_erg = []
for i in range(len(x)):
    erg = a[0] + a[1]*x[i] + a[2]*x[i]**2
    liste_erg.append(erg)
    z = np.array(liste_erg)
print(z)
#plt.plot(x, y, '-ob', x, z, '-*r')
#plt.show()

# Rechnen den Cost (z, y):
normes = np.abs(z-y)
e = np.mean(normes**2)
print(e)
