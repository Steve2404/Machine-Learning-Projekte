# Version N°1 der Eratosthenes Funktion
def Eratosthenes(N):
    liste_nbre_premier = [1]
    for n in range(N):
        if n > 1:
            cmpP = 1 + sum(n % i == 0 for i in range(2, n))
            if cmpP < 2:
                liste_nbre_premier.append(n)
    print(liste_nbre_premier)
Eratosthenes(100)






