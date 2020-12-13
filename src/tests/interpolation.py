##########################################################
#                      Problema 3)                       #
#Aluno: Victor Kaio                                      #
##########################################################


import numpy as np;
import cv2 as cv;
import os
from matplotlib import pyplot as plt

std_images_path = os.path.abspath('C:/Users/VK/Desktop/SAVE/7_semestre/PDI/test/')

#pegar a máscara da imagem
file_name = ('imss3.jpg')
file = os.path.join(std_images_path, file_name)
img = cv.imread(file,0)

#pegar a imagem com máscara retirada
img2 = cv.imread('imss.jpg',0)

#pegar a imagem original (apenas para comparação)
img3 = cv.imread('5.jpg',0)

#print imagem com a máscara retirada
cv.imshow('mascret',img2)

#print imagem original
cv.imshow('ori',img3)

#print da máscara
cv.imshow('masc',img)

#interpolação de acordo com a cor e com os bits ao lado (selecionado com a máscara)
dst = cv.inpaint(img2,img,3,cv.INPAINT_TELEA)

#print imagem reconstruída
cv.imshow('img-rec', dst)

#salvar o resultado
cv.imwrite('rec.jpg', dst)

cv.waitKey(0)
cv.destroyAllWindows()