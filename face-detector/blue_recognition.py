#Ejemplo de deteccion facial con OpenCV y Python
#Por Glare
#www.robologs.net

import numpy as np
import cv2

#cargamos la plantilla e inicializamos la webcam:
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)

while(True):
    #leemos un frame y lo guardamos
    ret, img = cap.read()

    # blurred = cv2.pyrMeanShiftFiltering(img, 31, 91)
    # gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # _, contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # cv2.drawContours(img, contours, -1, (0, 0, 255))

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    #Filtrar el ruido con un CLOSE/OPEN
    kernel = np.ones((6,6), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    #Difuminar la mascara para suavizar los contornos y aplicar filtro canny
    blur = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(mask, 1, 2)

    #Si el area blanca de la mascara es superior a 500px, no se trata de ruido
    _, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]

    i = 0
    for extension in areas:
        if extension > 600:
            actual = contours[i]
            approx = cv2.approxPolyDP(
                actual, 0.05 * cv2.arcLength(actual, True), True)
            if len(approx) == 4:
                cv2.drawContours(img, [actual], 0, (0, 0, 255), 2)
                cv2.drawContours(mask, [actual], 0, (0, 0, 255), 2)
            i = i + 1

    idx = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 50 and h > 50:
            idx += 1
            new_img = img[y:y + h, x:x + w]
            cv2.imwrite(str(idx) + '.png', new_img)

    cv2.imshow('mask', mask)
    cv2.imshow('Camara', img)

    #con la tecla 'q' salimos del programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2 - destroyAllWindows()
