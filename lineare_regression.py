# -*- coding: utf-8 -*-



import numpy as np

def extend_matrix(X):  
    # TODO: setze X_ext
    One = np.ones((X.shape[0], 1))
    X_ext = np.concatenate((One, X), axis=1)
    return X_ext



def LR_fit(X, y):
    # TODO: berechne theta
    Xe = extend_matrix(X)
    Xt = Xe.T
    theta = np.linalg.solve(Xt.dot(Xe), Xt.dot(y))
    return theta


def LR_predict(X, theta):
    # TODO: berechne y
    y = extend_matrix(X).dot(theta)
    return y
    

def r2_score(X, y, theta):
    # TODO: berechne r2
    r1 = LR_predict(X, theta)
    num = ((r1 - y) ** 2).sum()
    denum = ((y - np.mean(y))**2).sum()
    r2 = np.sqrt(1 - (num/denum))
    return r2


matriceX = np.array([[2, 2, 3],[0, 4, 1], [6, 2, 1]])
vektorY = np.array([1, 2, 1])

teta = LR_fit(matriceX, vektorY)
predict = LR_predict(matriceX, teta)
score = r2_score(matriceX, vektorY, teta)
print(score)