import cv2 as cv;
import numpy as np;
import matplotlib.pyplot as plt;
import collections
import os

## Calcular quantidade e porcentagem de cada componente encontrado
def qtd_porc(img):
    aux = np.asarray(img).reshape(-1)

    # Find pixel values and quantity of an image
    counter=collections.Counter(aux)
    counterKeys = list(counter.keys());
    counterValues = list(counter.values());
    
    # Size of vectors
    n = len(list(counter.keys()));
    
    # Array aux
    aux2 = [];
    
    # Calc porcentagem
    porcentagem = [round((x*100)/sum(counterValues),2) for x in counterValues];
    
    
    # Info
    for i in range(n):
        aux2.append({'label': counterKeys[i],
                     'qtd': counterValues[i],
                     'porc': porcentagem[i]});
    
    # Return info, keys, values(qtd), porcentagem
    return aux2, counterKeys, counterValues, porcentagem;

# Find position of bigger
def index_max_porc(labels):
    # Print qtd and porcentagem
    inteiro, keys, values, porc = qtd_porc(labels);
    porc[0] = 0;
    porc[1] = 0;
    n = porc.index(max(porc));
    
    return n;
    
# Find position of bigger
def index_max_porc2(labels, z):
    # Print qtd and porcentagem
    inteiro, keys, values, porc = qtd_porc(labels);
    porc[0] = 0;
    porc[1] = 0;
    porc[3] = 0;
    porc[10] = 0;
    porc[z] = 0;
    n = porc.index(max(porc));
    
    return n;

def create_mask(img):
    
    # Scaling and transforming for uint8
    img2 = cv.normalize(img, None, -1, 1, cv.NORM_MINMAX, cv.CV_8U);
    
    # Labeling
    num_labels, labels = cv.connectedComponents(img2);
    
    # Print qtd and porcentagem
    n = index_max_porc(labels);
    
    n2 = (index_max_porc2(labels, n))
    
    # Replace values in images
    labels1 = np.where(labels!=n, 0, labels);
    labels2 = np.where(labels!=n2, 0, labels);
    labels = ~labels1 * ~labels2;
    
    # # kernel = np.ones((50,50),np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(2,2))
    
    # Scaling and transforming for uint8
    img4 = cv.normalize(labels, None, -1, 255, cv.NORM_MINMAX, cv.CV_8U);   
        
    img4 = cv.dilate(img4,kernel,iterations = 1) ## Dialtar image
    
    return img4;


def apply_mask(img, mask):
    M, N = img.shape
    img5 = np.copy(img)
    img6 = np.copy(img)
    for i in range (0, M):
        for j in range(0,N):
            img5[i,j] = 0
            
    for i in range (0, M):
        for j in range(0,N):
            if mask[i,j] > 0:
                img6[i,j] = 0
                img5[i,j] = 255
                
    return img6;

# Tests
img = cv.imread('8.jpg', 0);  # Read image
mask = create_mask(img);                # Create mask
mask_img = apply_mask(img, mask);       # Apply mask

   

M, N = img.shape
img5 = np.copy(img)
for i in range (0, M):
    for j in range(0,N):
        img5[i,j] = 0
        
for i in range (0, M):
    for j in range(0,N):
        if mask[i,j] > 0:
            img[i,j] = 0
            img5[i,j] = 255
            
        
cv.imshow('fd',img)
cv.imshow('feg',img5)

cv.imwrite('imss.jpg',img)
cv.imwrite('imss2.jpg',img5)

#pegar a imagem original (apenas para comparação)
img115 = cv.imread('8.jpg',0)

#print imagem original
cv.imshow('imgrr',img115)

#interpolação de acordo com a cor e com os bits ao lado (selecionado com a máscara)
dst = cv.inpaint(img,img5,3,cv.INPAINT_TELEA)

#print imagem reconstruída
cv.imshow('img-rec', dst)

#salvar o resultado
cv.imwrite('rec.jpg', dst)

cv.waitKey(0)
cv.destroyAllWindows()
