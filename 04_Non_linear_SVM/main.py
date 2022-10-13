import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
from sklearn.svm import SVC


def kernel_lineal(x1, x2):
    return x1.dot(x2.T)


def kernel_gauss(x1, x2,gamma=0.5): # gamma=1
    md = distance_matrix(x1,x2)
    return np.exp((-gamma)*(md**2))


def kernel_poly(x1, x2, d=3):
    return (x1.dot(x2.T))**d

# Generació del conjunt de mostres
X, y = make_classification(n_samples=400, n_features=5, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1,
                           random_state=9)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Els dos algorismes es beneficien d'estandaritzar les dades
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)


#Entrenam una SVM linear (classe SVC)
svm = SVC(C=1.0, kernel='linear')
svm.fit(X_transformed, y_train)
y_predicted = svm.predict(X_test_transformed)

differences = (y_predicted - y_test)
errors = np.count_nonzero(differences)

print(f'Rati d\'acerts en el bloc de predicció: {(len(y_predicted)-errors)/len(y_predicted)}')

#Entrenam una SVM linear (classe SVC) amb el kernel lineal
svm1 = SVC(C=1.0, kernel=kernel_lineal)
svm1.fit(X_transformed, y_train)
y_predicted = svm1.predict(X_test_transformed)

differences = (y_predicted - y_test)
errors = np.count_nonzero(differences)

print(f'Rati d\'acerts en el bloc de predicció amb kernel lineal: {(len(y_predicted)-errors)/len(y_predicted)}')
print("Veim que el rati d'accerts es igual, la cosa està bé")
print("\n")

#Entrenam una SVM linear (classe SVC) amb el kernel gausia nostro
svm2 = SVC(C=1.0, kernel='rbf', gamma=1)
svm2.fit(X_transformed, y_train)
y_predicted = svm2.predict(X_test_transformed)

differences = (y_predicted - y_test)
errors = np.count_nonzero(differences)
print(f'Rati d\'acerts en el bloc de predicció amb kernel gauss llibreria: {(len(y_predicted)-errors)/len(y_predicted)}')

#Entrenam una SVM linear (classe SVC) amb el kernel gausia nostro
svm3 = SVC(C=1.0, kernel=kernel_gauss)
svm3.fit(X_transformed, y_train)
y_predicted = svm3.predict(X_test_transformed)

differences = (y_predicted - y_test)
errors = np.count_nonzero(differences)
print(f'Rati d\'acerts en el bloc de predicció amb kernel propi gauss: {(len(y_predicted)-errors)/len(y_predicted)}')
print("\n")


#Entrenam una SVM linear (classe SVC) pollinomic de python
svm4 = SVC(C=1.0, kernel='poly', gamma=1, degree=3)
svm4.fit(X_transformed, y_train)
y_predicted = svm4.predict(X_test_transformed)

differences = (y_predicted - y_test)
errors = np.count_nonzero(differences)
print(f'Rati d\'acerts en el bloc de predicció amb kernel polinomic de python: {(len(y_predicted)-errors)/len(y_predicted)}')

#Entrenam una SVM linear (classe SVC) pollinomic de nostre
svm5 = SVC(C=1.0, kernel=kernel_poly)
svm5.fit(X_transformed, y_train)
y_predicted = svm5.predict(X_test_transformed)

differences = (y_predicted - y_test)
errors = np.count_nonzero(differences)
print(f'Rati d\'acerts en el bloc de predicció amb kernel polinomic nostre: {(len(y_predicted)-errors)/len(y_predicted)}')
print("\n")


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

poly = PolynomialFeatures(3)
X_transformed2 = poly.fit_transform(X_train)
#y_transformed2 = poly.fit_transform(y_train)

#Entrenam una SVM linear (classe SVC)
svm = SVC(C=1.0, kernel='linear')
svm.fit(X_transformed2, y_train)
y_predicted = svm.predict(X_transformed2)

differences = (y_predicted - y_train)
errors = np.count_nonzero(differences)

print(f'Rati d\'acerts en el bloc de predicció: {(len(y_predicted)-errors)/len(y_predicted)}')


