# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 18:20:15 2016

@author: Samy
"""
from PIL import Image, ImageOps
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from time import time

#• définition de la taille des blocs (patchs)


# prend une image en entrée et renvoie un tableau de niveau de gris
img = Image.open("roof_images\\38128663.jpg")

def imgToGrayArray(img, resize=False, shape=img.size):
    if resize:
        img = img.resize(shape, 0)

    img = ImageOps.grayscale(img)
    img2 = np.asarray(img).astype(np.float)
    
    return(img2)

img2 = imgToGrayArray(img)

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
data = constructPatches(img2, patch_size)
t1 = time() - t0
print('temps d\'extraction: %.2fs' % t1)
# il faut n_components > ncol des images 
# initialisation d'un dictionnaire
# n_components: taille du dictionnaire
# on fit le dictionnaire sur l'image de base normalisée

print('construction du dictionnaire et fit sur les data')
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=2*img2.shape[1], alpha=1, n_iter=100, n_jobs=-1)#, transform_algorithm='omp')
V = dico.fit(data).components_
t1 = t0 - time()
print('temps fit dico: %.fs ' % t1)

"""
# on enleve le bruit que l'on suppose gaussien
data = data = extract_patches_2d(img, patch_size)
data = data.reshape(data.shape[0], -1)
intercept = np.mean(data, axis=0)
data -= intercept
"""

# définition des algos de transformations (OMP avec 1 et 2 atomes, LAR regression 5 atomes, et autre chose )
transform_algorithms = [('omp1', 'omp',{'transform_n_nonzero_coefs': 10})]
"""
,
('Least-angle regression\n5 atoms', 'lars',
{'transform_n_nonzero_coefs': 5}),
('Thresholding\n alpha=0.1', 'threshold', {'transform_alpha': .1})
"""
    

#} création de plusieurs images reconstruites stockées dans un dictionnaire
reconstructions = {}
for title, transform_algorithm, kwargs in transform_algorithms:
    reconstructions[title] = img2.copy()
    dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
    print('calcul de la transformation des data dans le dico')
    t0 = time()
    code = dico.transform(data)
    t1 = t0 - time()
    print('temps transformation data dans dico: %.fs' % t1)
    patches = np.dot(code, V)
    patches = patches.reshape(len(data), *patch_size)
    # on enleve les img.size[1]//2 premieres colonnes 
    #reconstructions[title][:, img.size[1]//2:] = reconstruct_from_patches_2d(patches, (img.size[0], img.size[1]//2))
    print('reconstruction de l\'image par les patchs')
    t0 = time()    
    reconstructions[title] = reconstruct_from_patches_2d(patches, img2.shape)
    t1 = t0 - time()
    print('temps reconstruction par patches: %.fs' % t1)

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


