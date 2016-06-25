# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 15:55:46 2016

@author: Zied
"""

from PIL import Image
im = Image.open("C:\\Users\\TOSHIBA\\Desktop\\3A\\datasciencegame\\roof_images\\-215031.jpg")
im.show()

im = np.asarray(im) 
im[0,0,0] #48
im[0,0,1] #37
im[0,0,2] #36

for i in range(1):
    
im.getpixel((0,0))

list = im.getpixel(())