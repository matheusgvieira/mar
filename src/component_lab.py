import cv2 as cv;
import numpy as np;
import matplotlib.pyplot as plt;
import collections

## Calcular quantidade e porcentagem de cada componente encontrado
def qtd_porc(img):
    aux = np.asarray(img).reshape(-1)

    # Find pixel values and quantity of an image
    counter=collections.Counter(aux)
    counterKeys = list(counter.keys());
    counterValues = list(counter.values());
    
    n = len(list(counter.keys()));
    
    aux2 = [];
    
    # print(f'counter = {counter}')
    # print(f'keys = {counterKeys}')
    # print(f'values = {counterValues}')
    
    porcentagem = [round((x*100)/sum(counterValues),2) for x in counterValues];
    
    # print(f'porcentagem = {porcentagem}')
    
    for i in range(n):
        aux2.append({'label': counterKeys[i],
                     'qtd': counterValues[i],
                     'porc': porcentagem[i]});
    
    return aux2, counterKeys, counterValues, porcentagem;

def index_max_porc(labels):
    # Print qtd and porcentagem
    inteiro, keys, values, porc = qtd_porc(labels);
    porc[0] = 0;
    n = porc.index(max(porc));
    
    return n;

# Read Image
img = cv.imread('../images/1.jpg', 0);

# Scaling and transforming for uint8
img2 = cv.normalize(img, None, -1, 1, cv.NORM_MINMAX, cv.CV_8U);

# Labeling
num_labels, labels = cv.connectedComponents(img2);

# Print qtd and porcentagem
n = index_max_porc(labels);

# Replace values in images
labels = np.where(labels>n, 0, labels);

# Scaling and transforming for uint8
img3 = cv.normalize(labels, None, -1, 1, cv.NORM_MINMAX, cv.CV_8U);

# kernel = np.ones((50,50),np.uint8)
kernel = cv.getStructuringElement(cv.MORPH_RECT,(9,9))
erosion = cv.erode(img3,kernel,iterations = 1)

# Labeling
num_labels, labels = cv.connectedComponents(erosion);

n = index_max_porc(labels);

# Replace values in images
labels = np.where(labels!=n, 0, labels);

# Scaling and transforming for uint8
img4 = cv.normalize(labels, None, -1, 255, cv.NORM_MINMAX, cv.CV_8U);

# img4 = cv.dilate(img4,kernel,iterations = 1)

plt.figure()
plt.axis("off")
plt.imshow(img4, cmap=plt.cm.gray)
plt.show()

cv.imwrite('../outputs/label4.png', img4);

# M, N = img.shape
# img5 = np.copy(img)
# for i in range (0, M):
#     for j in range(0,N):
#         img5[i,j] = 0
        
# for i in range (0, M):
#     for j in range(0,N):
#         if img4[i,j] > 0:
#             img[i,j] = 0
#             img5[i,j] = 255
        
# cv.imshow('fd',img)
# cv.imshow('feg',img5)

# cv.imwrite('../outputs/imss.jpg',img)
# cv.imwrite('../outputs/imss2.jpg',img5)
            
# cv.waitKey(0)
# cv.destroyAllWindows()