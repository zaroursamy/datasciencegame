# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 18:20:15 2016

@author: Samy
"""

import os
from PIL import Image, ImageOps
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from time import time

# chemin
path = 'C:\\Users\\Samy\\datasciencegame'
os.chdir(path)

# prend une image en entrée et renvoie un tableau de niveau de gris
img = Image.open("roof_images\\40437685.jpg")

def imgToGrayArray(img, resize=False, shape=img.size):
    if resize:
        img = img.resize(shape, 0)

    img = ImageOps.grayscale(img)
    img2 = np.asarray(img).astype(np.float)
    
    return(img2)

img2 = imgToGrayArray(img, True, (200,200))

# taille des patchs (les prendre assez petits pour apprendre)
patch_size = tuple(map(lambda x: x//5, img2.shape))

# reforme une image en une collection de patches, puis la normalise
def constructPatches(img2, patch_size, scale=True):   
    data = extract_patches_2d(img2, patch_size)
    data = data.reshape(data.shape[0], -1)
    
    if scale:
        data -= np.mean(data, axis=0)
        data /= np.std(data, axis=0)
    
    return(data)

print('extraction des patches')
t0 = time()
data = constructPatches(img2, patch_size, False)
t1 = time() - t0
print('temps d\'extraction: %.2fs' % t1)

print('construction du dictionnaire et fit sur les data')
# il faut n_components > ncol des images 
# initialisation d'un dictionnaire
# n_components: taille du dictionnaire
# on fit le dictionnaire sur l'image de base normalisée
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=2*img2.shape[1], alpha=1, n_iter=100)
V = dico.fit(data).components_
t1 = time() - t0
print('temps fit dico: %.fs ' % t1)

# définition des algos de transformations (OMP avec 1 et 2 atomes, LAR regression 5 atomes, et autre chose )
transform_algorithms = [('omp10', 'omp',{'transform_n_nonzero_coefs': 10}), ('omp5', 'omp',{'transform_n_nonzero_coefs': 5})]

#} création de plusieurs images reconstruites stockées dans un dictionnaire
def reconstructImages(transform_algorithms):
    reconstructions = {}
    
    for title, transform_algorithm, kwargs in transform_algorithms:
        reconstructions[title] = img2.copy()
        dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
        code = dico.transform(data)
        patches = np.dot(code, V)
        patches = patches.reshape(len(data), *patch_size)
        reconstructions[title] = reconstruct_from_patches_2d(patches, img2.shape)
        
    return(reconstructions)


print('reconstruction image')
t0 = time()
reconstructions = reconstructImages(transform_algorithms)
t1 = time() - t0
print('temps reconstruction image: %.fs' %t1)

