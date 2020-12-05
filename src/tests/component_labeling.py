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
    
    print(f'counter = {counter}')
    print(f'keys = {counterKeys}')
    print(f'values = {counterValues}')
    
    porcentagem = [round((x*100)/sum(counterValues),2) for x in counterValues];
    
    print(f'porcentagem = {porcentagem}')
    
    for i in range(n):
        aux2.append({'label': counterKeys[i],
                     'qtd': counterValues[i],
                     'porc': porcentagem[i]});
    
    return aux2;

# Read Image
img = cv.imread('../images/Fig0539(c)(shepp-logan_phantom).tif', 0);

# Replace values in images
Img = np.where(img>0, 255, img);

# Labeling
num_labels, labels = cv.connectedComponents(~Img);

# Replace values in images
labels2 = np.where(labels==1, 0, labels);
labels2 = np.where(labels2 > 0, 1, labels2);

# Scaling and transforming for uint8
img2 = cv.normalize(labels2, None, -1, 1, cv.NORM_MINMAX, cv.CV_8U);

# Labeling
num_labels, labels = cv.connectedComponents(img2);
print(f'Número de componentes rotulados = {num_labels - 1}');

# Print qtd and porcentagem
print(qtd_porc(labels));

# Print images
fig, axes = plt.subplots(2, 2,figsize=(10, 10));
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray);
ax[0].set_title('Image Original');
ax[1].imshow(~Img, cmap=plt.cm.gray)
ax[1].set_title('Binary Image Inverted');
ax[2].imshow(labels, cmap=plt.cm.gray)
ax[2].set_title('Labeled Image ');
ax[3].imshow(cv.bitwise_or(img, ~img2), cmap=plt.cm.gray)
ax[3].set_title('Subtraído');

for a in ax:
    a.axis('off')
fig.tight_layout()
plt.show() 