import cv2 


cap = cv2.VideoCapture(0)


while True:

    suc, img = cap.read()
    cv2.imshow("Veb",  cv2.flip(img, 1))


    key = cv2.waitKey(1)

    if key == ord('q') or key == 27:
        break