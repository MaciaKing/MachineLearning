import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(10, 10),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)


def aplanarCorba():
    #fer un for sobre "pca.explained_variance_ratio_"
    None

digits = datasets.load_digits()
#plot_digits(digits.data[:100, :])

# Devallam les dimensions del dataset
pca = PCA()
pca.fit(digits.data)
data_t = pca.transform(digits.data)
#plt.plot(pca.explained_variance_ratio_) # Primer plot profe
plt.bar(np.arange(0, np.size(pca.explained_variance_ratio_) ), np.cumsum(pca.explained_variance_ratio_))
plt.show()

#plt.plot([1,10],pca.explained_variance_ratio_)
#plt.show()
# Decdim  0.95

'''
pca1 = PCA(n_components=10)  # Valor del primer plot
pca1.fit(digits.data)
#gm = GaussianMixture(random_state=0).bic(digits.data)
gm = GaussianMixture(random_state=0).fit(digits.data)
#plt.plot([1, 10], pca1.explained_variance_ratio_)
plt.plot(np.arange(1, np.size(pca1.explained_variance_ratio_) ), np.cumsum(pca1.explained_variance_ratio_))

plt.show()
'''