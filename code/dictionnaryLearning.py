# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 18:20:15 2016

@author: Samy
"""
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import misc
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version

#• définition de la taille des blocs (patchs)


# test sur une image en niveaux de gris
img = Image.open("roof_images\\38128663.jpg")
img = ImageOps.grayscale(img)
img2 = np.asarray(img)
img2 = img2.astype(np.float)

# taille des patchs (les prendre assez petits pour apprendre)
patch_size = (5,5)

# reforme une image en une collection de patches, puis la normalise
data = extract_patches_2d(img2, patch_size)
data = data.reshape(data.shape[0], -1)
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)

# il faut n_components > ncol des images 
# initialisation d'un dictionnaire
# n_components: taille du dictionnaire
# on fit le dictionnaire sur l'image de base normalisée
dico = MiniBatchDictionaryLearning(n_components=1000, alpha=1, n_iter=500)
V = dico.fit(data).components_

"""
# on enleve le bruit que l'on suppose gaussien
data = data = extract_patches_2d(img, patch_size)
data = data.reshape(data.shape[0], -1)
intercept = np.mean(data, axis=0)
data -= intercept
"""

# définition des algos de transformations (OMP avec 1 et 2 atomes, LAR regression 5 atomes, et autre chose )
transform_algorithms = [('omp1', 'omp',{'transform_n_nonzero_coefs': 1}),('omp2', 'omp',{'transform_n_nonzero_coefs': 2})]
"""
,
('Least-angle regression\n5 atoms', 'lars',
{'transform_n_nonzero_coefs': 5}),
('Thresholding\n alpha=0.1', 'threshold', {'transform_alpha': .1})
"""
    

#} reconstruction = image reconstruite
reconstructions = {}
for title, transform_algorithm, kwargs in transform_algorithms:
    reconstructions[title] = img
    dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
    code = dico.transform(data)
    patches = np.dot(code, V)

    """if transform_algorithm == 'threshold':
        patches -= patches.min()
        patches /= patches.max()

    #patches += intercept
    patches = patches.reshape(len(data), *patch_size)
    if transform_algorithm == 'threshold':
        patches -= patches.min()
        patches /= patches.max()
        reconstructions[title][:, width // 2:] = reconstruct_from_patches_2d(
        patches, (height, width // 2))"""


