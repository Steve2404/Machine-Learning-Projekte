# -*- coding: utf-8 -*-

# logistische_regression
#
# Routinen zur Berechnung der bivariaten logistischen Regression
# mit Modellfunktion
#
#   h_theta(x) = sigma(theta_0 + theta_1 * x_1 + theta_2 * x_2)
#
# mit
#
#   sigma(t) = 1/(1+exp(-t))
#
# und Kostenfunktion
# 
#   J(theta) = -1/m sum_(i=1)^m (y^(i) log(h_theta(x^(i))) 
#                               + (1-y^(i)) log(1 - h_theta(x^(i))) 
#
# Der Vektor theta wird als
#
#   (theta_0, theta_1, theta_2)
#
# gespeichert. Die Feature-Matrix mit m Daten und n Features als
#
#       [ - x^(1) - ]
#   X = [    .      ]    (m Zeilen und n Spalten)
#       [ - x^(m) - ]
#

import numpy as np

#%% extend_matrix (vom letzten Mal verwenden, wird nicht geprüft)

# Erweitert eine Matrix um eine erste Spalte mit Einsen
#
# X_ext = extend_matrix(X)
#
# Eingabe:
#   X      Matrix m x n (numpy.ndarray)
#
# Ausgabe
#   X_ext  Matrix m x (n+1) der Form [1 X] (numpy.ndarray)
#
def extend_matrix(X):  # sourcery skip: inline-immediately-returned-variable
    One = np.ones((X.shape[0], 1))
    X_ext = np.concatenate((One, X), axis=1)
    return X_ext



#%% logistic_cost_function

# Berechnung der Kostenfunktion der logistischen Regression und deren 
# Gradienten
#
# J, Jgrad = logistic_cost_function(X,y, theta)
#
# Eingabe:
#   X      Matrix m x 2 mit m Datenpunkten und 2 Features (numpy.ndarray)
#   y      Vektor der Länge m der Zielwerte (numpy.ndarray)
#   theta  Vektor der  Länge 3 der Parameter (numpy.ndarray)
#
# Ausgabe
#   J      Wert der regularisierten Kostenfunktion (Skalar)
#   Jgrad  Gradient der regularisierten Kostenfunktion (numpy.ndarray)
#
def logistic_cost_function(X,y, theta):
    t = extend_matrix(X).dot(theta)
    h_theta = 1/(1 + np.exp(-t))
    J = -np.mean(y.dot(np.log(h_theta)) + (1-y).dot(np.log(1- h_theta))).sum()
    Jgrad = np.mean(h_theta - y)*(extend_matrix(X))
    
    return J, Jgrad 

       
X = np.array([[1,2],
              [3,1],
              [3,4]])
Y = np.array([1,0,0])
theta = np.array([1, 2, 0])
cost = logistic_cost_function(X, Y, theta)
#print(f" cost fonktion: {cost[0]} und gradient : {cost[1]}")



#%% LogisticRegression_fit

# Berechnung der optimalen Parameter der multivariaten logistischen Regression 
# mithilfe des Gradientenabstiegsverfahrens
#
# theta, J = LogisticRegression_fit(X,y,eta,tol)
#
# Die Iteration soll abgebrochen werden, falls 
#
#   || grad J || < tol
#
# gilt, wobei ||.|| die (euklidsche) Länge eines Vektors ist. Die Iteration
# soll abbrechen (mittels raise), falls die Kostenfunktion nicht fällt. Als
# Startvektor soll der Nullvektor gewählt werden.
#
# Eingabe:
#   X      Matrix m x 2 mit m Datenpunkten und 2 Features (numpy.ndarray)
#   y      Vektor der Länge m der Zielwerte (numpy.ndarray)
#   eta    Learning rate (Skalar)
#   tol    Toleranz der Abbruchbedingung
#
# Ausgabe
#   theta   Aktueller Vektor der Länge 3 der optimalen Parameter (numpy.ndarray)
#   J       Aktueller Wert der Kostenfunktion (float)
#
def LogisticRegression_fit(X,y,eta,tol):
    theta = np.zeros(3)
    J = 0
    i=0
    while (np.linalg.norm(logistic_cost_function(X, y, theta)[1])> tol):
        try:
            
            theta = theta - eta*(logistic_cost_function(X, y, theta)[1])
            J = logistic_cost_function(X,y,theta)[0]
            print(f"Les valeur de Theta: {theta[0]} J={J}")
            #Ici on doit comparer le valeur de l ancien J avec la nouvelle valeur, si elle est grande alors on arrete tout avec raise.
        except Exception as ex:
            raise ex
    return theta, J

fit = LogisticRegression_fit(X, Y, 10, 1e-4)
print(f"neue Theta: {fit[0]}")

#%% LogisticRegression_predict

# Berechnung der Vorhersage der multivariaten logistischen Regression
#
# y = LogisticRegression_predict(X,theta)
#
# Eingabe:
#   X      Matrix m x 2 mit m Datenpunkten und 2 Features (numpy.ndarray)
#   theta  Vektor der  Länge 3 der Parameter (numpy.ndarray)
#
# Ausgabe
#   y      Vektor der Länge m der Vorhersagewerte (numpy.ndarray)
#
#
def LogisticRegression_predict(X, theta):
    # sourcery skip: inline-immediately-returned-variable
    t = extend_matrix(X).dot(theta)
    h_theta = 1/(1 + np.exp(-t))
    y = 1 if h_theta >= 0.5 else 0
    return y
    

#%% accuracy_score

# Berechnung der Genauigkeit
#
# acc = accuracy_score(y_true,y_pred)
#
# Eingabe:
#   y_true Vektor der Länge m der wahren Werte (numpy.ndarray)
#   y_pred Vektor der Länge m der Vorhersagewerte (numpy.ndarray)
#
# Ausgabe
#   acc    Genauigkeit (Skalar)
#
def accuracy_score(y_true,y_pred):
    # sourcery skip: inline-immediately-returned-variable
    anzahl_richtig = len(y_true == y_pred)
    acc = anzahl_richtig/len(y_pred)
    return acc

