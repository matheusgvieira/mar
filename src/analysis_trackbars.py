import cv2 as cv
import numpy as np

def nothing(x):
    pass

# carragar a imagem para seleção manual
image = cv.imread('5.jpg')
# Create a window
cv.namedWindow('image')

# criação das trackbars para a análise
cv.createTrackbar('HMin','image',0,179,nothing)
cv.createTrackbar('SMin','image',0,255,nothing)
cv.createTrackbar('VMin','image',0,255,nothing)
cv.createTrackbar('HMax','image',0,179,nothing)
cv.createTrackbar('SMax','image',0,255,nothing)
cv.createTrackbar('VMax','image',0,255,nothing)

# Colocar os padrões de valores máximos nas trackbars HSV
cv.setTrackbarPos('HMax', 'image', 179)
cv.setTrackbarPos('SMax', 'image', 255)
cv.setTrackbarPos('VMax', 'image', 255)

# Inicializar para verificar os valores de HSV minimo e maximo
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

output = image
wait_time = 33

while(1):

    # Pega a localição atual da imagem para as trackbars
    hMin = cv.getTrackbarPos('HMin','image')
    sMin = cv.getTrackbarPos('SMin','image')
    vMin = cv.getTrackbarPos('VMin','image')

    hMax = cv.getTrackbarPos('HMax','image')
    sMax = cv.getTrackbarPos('SMax','image')
    vMax = cv.getTrackbarPos('VMax','image')

    # Setar o minimo e o máximo das trackbars no display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # criar a imagem HSV e setar um range dela
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    output = cv.bitwise_and(image,image, mask= mask)

    # Print da imagem no display
    if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # saída da imagem selecionada
    cv.imshow('image',output)

    # Salvar a imagem e sair do display quando a tecla 'q' for apertada
    if cv.waitKey(wait_time) & 0xFF == ord('q'):
        cv.imwrite('imss3.jpg',output)
        break

cv.destroyAllWindows()