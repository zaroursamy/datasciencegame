
# coding: utf-8

# imports
import os
import numpy as np
#from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image, ImageOps
from sklearn.cross_validation import train_test_split

# chemin
path = 'C:\\Users\\Samy\\datasciencegame'
os.chdir(path)

# lecture du csv d'apprentissage
apprentissage = pd.read_csv('id_train.csv', sep=";")
apprentissage = apprentissage.astype(np.str)

# fonction qui importe les images dans des listes et split selon train/test
def loadImages(train): 
    app = list(map(lambda x: x+".jpg", list(train.Id)))
    
    # liste qui contient les images train
    listTrain = []
    for img in app:
       listTrain.append(Image.open('roof_images\\'+img))

    return(listTrain)

listTrain = loadImages(apprentissage)
# fonction qui split en app/test. alpha=%du test
def splitTrainTest(listeImages, alpha):
    listTrain, listTest = train_test_split(listeImages, test_size=alpha, random_state=50)
    return(listTrain, listTest)
    
train, test = splitTrainTest(listTrain, 0.3)
# remarque: on accède au nom d'une image via img.filename

