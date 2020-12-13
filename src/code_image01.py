import cv2 as cv;
import numpy as np;
import matplotlib.pyplot as plt;
import collections

def fg(img, img2):   
    ret,thresh1 = cv.threshold(img,245,255,cv.THRESH_BINARY)
    img = thresh1
            
    N, M = img.shape
    for i in range(0, N):
        for j in range(0, M):
            if (img[i,j]==255):
                img2[i,j] = 220
            if (img2[i,j]<20):
                img2[i,j] = 0
    
    return img2
    

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
    
    print(f'porcentagem = {porcentagem}')
    
    for i in range(n):
        aux2.append({'label': counterKeys[i],
                     'qtd': counterValues[i],
                     'porc': porcentagem[i]});
    
    return aux2, counterKeys, counterValues, porcentagem;

    # Read Image
def im(img, r):            
    # Scaling and transforming for uint8
    img2 = cv.normalize(img, None, -1, 1, cv.NORM_MINMAX, cv.CV_8U);
    
    # Labeling
    num_labels, labels = cv.connectedComponents(img2);
    
    # Print qtd and porcentagem
    inteiro, keys, values, porc = qtd_porc(labels);
    print(porc.index(max(porc)));
    
    # Replace values in images
    if (r==1):
        for i in range(15,17):
            labels = labels + np.where(labels>i, 0, labels);
    
        labels = labels + np.where(labels>12, 0, labels);
    
    if (r==2):
        labels = np.where(labels>10, 0, labels);
        
    if (r==3):
        for i in range(1,5):
            labels = labels + np.where(labels>i, 0, labels);
    
        labels = labels + np.where(labels>12, 0, labels);
        
    if (r==4):
        for i in range(1,8):
            labels = labels + np.where(labels>i, 0, labels);
    
        labels = labels + np.where(labels>12, 0, labels);
    
    # Scaling and transforming for uint8
    img3 = cv.normalize(labels, None, -1, 1, cv.NORM_MINMAX, cv.CV_8U);
    
    # kernel = np.ones((50,50),np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(7,7))
    erosion = cv.erode(img3,kernel,iterations = 1)
    
    # Labeling
    num_labels, labels = cv.connectedComponents(erosion);
    
    # Print qtd and porcentagem
    # print(qtd_porc(labels));
    
    # Replace values in images
    labels = np.where(labels!=2, 0, labels);
    
    # Scaling and transforming for uint8
    img4 = cv.normalize(img3, None, -1, 255, cv.NORM_MINMAX, cv.CV_8U);
    
    return img4


img = cv.imread('5.jpg', 0); 

img2 = im(img, 1)

plt.figure()
plt.axis("off")
plt.imshow(img2, cmap=plt.cm.gray)
plt.show()

M, N = img.shape
img3 = np.copy(img)
for i in range (0, M):
    for j in range(0,N):
        img3[i,j] = 0
        
for i in range (0, M):
    for j in range(0,N):
        if img2[i,j] > 0:
            img[i,j] = 0
            img3[i,j] = 255
        
cv.imshow('fd',img)
cv.imshow('feg',img3)

cv.imwrite('imss.jpg',img)
cv.imwrite('imss2.jpg',img3)

#interpolação de acordo com a cor e com os bits ao lado (selecionado com a máscara)
dst = cv.inpaint(img,img3,3,cv.INPAINT_TELEA)

#print imagem reconstruída
cv.imshow('img-rec', dst)

#salvar o resultado
cv.imwrite('rec.jpg', dst)

img7 = np.copy(dst) 

img5 = im(img, 2)

plt.figure()
plt.axis("off")
plt.imshow(img5, cmap=plt.cm.gray)
plt.show()

M, N = img7.shape
img6 = np.copy(img7)
for i in range (0, M):
    for j in range(0,N):
        img6[i,j] = 0
        
for i in range (0, M):
    for j in range(0,N):
        if img5[i,j] > 0:
            img7[i,j] = 0
            img6[i,j] = 255
        
cv.imshow('fd2',img7)
cv.imshow('feg2',img6)

cv.imwrite('imss3.jpg',img7)
cv.imwrite('imss4.jpg',img6)


#interpolação de acordo com a cor e com os bits ao lado (selecionado com a máscara)
dst2 = cv.inpaint(img7,img6,3,cv.INPAINT_TELEA)

#print imagem reconstruída
cv.imshow('imgc', dst2)

#salvar o resultado
cv.imwrite('rec2.jpg', dst2)


img2 = np.copy(dst2)

f = np.copy(dst2).astype(np.float64)
zp = np.log(f+1)

Z = np.fft.fft2(zp)


g = np.fft.fftshift(Z)
t = 20*np.log(np.abs(g))

plt.imshow(t, cmap = 'gray')
plt.title('Espectro da imagem'), plt.xticks([]), plt.yticks([])
plt.show()

m,n = f.shape
D = np.copy(img).astype(np.float64)
H = np.copy(img).astype(np.float64)
n1 = np.floor(2*m+1)
n2 = np.floor(2*n+1)
for i in range(m):
    for j in range(n):
        D[i][j]=(((i-n1/2)**2+(j-n2/2)**2))**0.5

      
rL = 0.25
rH = 2
c =1
d0 = 20
 
for i in range(m):
    for j in range(n):
        H[i][j] = ((rH - rL)*(1-np.exp(-c*(((D[i][j]))/(d0**2)))) + rL)


gu = np.fft.fftshift(H)      
E = 20*np.log(np.abs(gu))

plt.imshow(E, cmap = 'gray')
plt.title('Espectro do filtro'), plt.xticks([]), plt.yticks([])
plt.show()
        
plt.plot(H);
plt.title("Filtro no domínio da frequência.")
plt.show();
  

S = H*Z
sp = np.fft.ifft2(S) 
g = np.real(np.exp(sp))

#ll=g.astype(np.uint8)
ll=cv.normalize(g, None, -1, 255, cv.NORM_MINMAX, cv.CV_8U)
cv.imshow('filt-n',ll)

cv.imwrite('rec3.jpg',ll)

hh = im (dst2,3)
cv.imshow('ieet', hh)

img8 = fg(hh,ll)

cv.imshow('ifg', img8)

hh2 = im (dst2,4)
cv.imshow('ieet2', hh2)

img9 = fg(hh2,img8)

cv.imshow('ifg2', img9)

cv.imwrite('rec4.jpg', img9)

img = cv.imread('5.jpg',0)
cv.imshow('oriii', img)
            
cv.waitKey(0)
cv.destroyAllWindows()