import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Adaline import Adaline
from sklearn.svm import SVC


# Generació del conjunt de mostres
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=2,
                           random_state=9)
y[y == 0] = -1

# Els dos algorismes es beneficien d'estandaritzar les dades
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X)

# Entrenam un perceptron
perceptron = Adaline(eta=0.0005, n_iter=60)
perceptron.fit(X_transformed, y)
#y_prediction = perceptron.predict(X) no hago prediccion solo quiero la linea dibujada

# TODO
#X_trainFit = scaler.fit_transform(X_transformed)

#Entrenam una SVM linear (classe SVC)
perceptronSVC = SVC(C=0.1, kernel='linear') # Linear Kernel, C--> quan més alt, mes tolerant es a fallar.
perceptronSVC.fit(X_transformed, y)
#perceptronSVC.predict(X_test) no quiero predecir nada, solo dibujarlo


plt.figure(1)

#  Mostram els resultats Adaline
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)
m = -perceptron.w_[1] / perceptron.w_[2]  #Calcula la pendiente
origen = (0, -perceptron.w_[0] / perceptron.w_[2])
plt.axline(xy1=origen, slope=m, c="blue", label="Adaline")


#  Mostram els resultats SVM
# TODO
m = -perceptronSVC.coef_[0][0] / perceptronSVC.coef_[0][1] #Calcula la pendiente
origen = (0, -perceptronSVC.intercept_[0] / perceptronSVC.coef_[0][1])
plt.axline(xy1=origen, slope=m, c="red", label="SVM")
plt.scatter( perceptronSVC.support_vectors_[:,0], perceptronSVC.support_vectors_[:,1], facecolors="none", edgecolors="green")


plt.legend()
plt.show()
