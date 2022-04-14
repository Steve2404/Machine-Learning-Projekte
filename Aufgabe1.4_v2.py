# Version N°2 der Eratosthenes Funktion
def Eratosthenes2(N):
    liste = list(range(1, N))
    for i in range(1, len(liste)):
        for j in range(i+1, len(liste)):
            c =  liste[i]*j
            if c in liste:
                liste.remove(c)
    print(liste)
Eratosthenes2(100)