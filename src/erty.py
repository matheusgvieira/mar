##########################################################
#                      Problema 3)                       #
#Aluno: Victor Kaio                                      #
##########################################################


import numpy as np;
import cv2 as cv;
import os
from matplotlib import pyplot as plt
    

std_images_path = os.path.abspath('C:/Users/VK/Desktop/SAVE/7_semestre/PDI/test/')

file_name = ('poij.jpeg')
file = os.path.join(std_images_path, file_name)
img = cv.imread(file,0)

img2 = cv.imread('imss.jpg',0)

img3 = cv.imread('1.jpg',0)

cv.imshow('ori',img2)

cv.imshow('oi',img3)

cv.imshow('fil',img)

dst = cv.inpaint(img2,img,3,cv.INPAINT_TELEA)

cv.imshow('img-rec', dst)

cv.imwrite('rec.jpg', dst)

cv.waitKey(0)
cv.destroyAllWindows()