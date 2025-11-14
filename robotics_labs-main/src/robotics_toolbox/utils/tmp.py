import cv2
import numpy as np
#TOTO je pouze testovaci soubor na nastaveni parametru HoughtCircles 
#THIS is just test file for settings of HoughtCircles param

image_path = '../hoop_grid_image_16.png'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

mask_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
mask_img = cv2.medianBlur(mask_img, 9) 

circles = cv2.HoughCircles(
    mask_img, cv2.HOUGH_GRADIENT,
    dp=1.2
    , minDist=50,
    param1=400,
    param2=30,  
    minRadius=0,
    maxRadius=0)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for x, y, r in circles[0, :]:
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)       
        cv2.circle(image, (x, y), 2, (0, 0, 255), 3)       

if circles is not None:
    circles = np.uint16(np.around(circles))  
    x, y, r = circles[0, 0]                  
    cv2.circle(image, (x, y), r, (0, 0 , 255), 2)  
    cv2.circle(image, (x, y), 2, (0, 0, 255), 3)  
    print(f"Nalezen kruh: střed=({x}, {y}), poloměr={r}")




cv2.namedWindow('Hoop Image', cv2.WINDOW_NORMAL)
cv2.imshow('Hoop Image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.imshow('mask', mask_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
