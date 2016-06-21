# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 16:07:23 2016

@author: Samy
"""
from funct.ools import reduce

# retourne le max de la ligne et de la colonne 
def maxOfImages(listeImages):
    # max des lignes
    mappingLigne = map(lambda x: x.size[0], listeImages)
    reducingLigne = reduce(lambda x,y: max(x,y), mappingLigne)
    
    # max des colonnes
    mappingCol =  map(lambda x: x.size[1], listeImages)
    reducingCol = reduce(lambda x,y: max(x,y), mappingCol)
    
    return(reducingLigne, reducingCol)
    
# redimmensionne les images au max
def reshapeWithoutBlack(listeImages):
    maxLigne, maxCol = maxOfImages(listeImages)
    listeImagesReshaped = list(map(lambda x: x.resize((maxLigne, maxCol), 0),listeImages))
    
    return(listeImagesReshaped)
    