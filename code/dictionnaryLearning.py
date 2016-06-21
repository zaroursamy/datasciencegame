# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 18:20:15 2016

@author: Samy
"""

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


# 
img = Image.open("roof_images\\-1191173.jpg")
img = np.asarray(img)
patch_size = (10,10)
data = extract_patches_2d(img, patch_size)

# il faut n_components > ncol des images 
# initialisation d'un dictionnaire
# n_components: taille du dictionnaire
dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500)

V = dico.fit(data)
