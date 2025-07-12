import cv2
import imutils

cam=cv2.VideoCapture(0)
firstFrame=None
area=500

while True:
    ret, img = cam.read()

    if not ret:
        print("Error: Couldn't read frame from the camera.")
        break  # Exit the loop or handle the error as needed

    img = imutils.resize(img, width=1000)
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussianImg = cv2.GaussianBlur(grayimg, (21, 21), 0)
    if firstFrame is None:
        firstFrame= gaussianImg
        continue
        
imgDiff=cv2.absdiff(firstFrame,gaussianimg)
threshImg=cv2.threshold(imgDiff,25,255,cv2.THRESH_BINARY)[1]
threshImg=cv2.dilate(threshImg,None,iterations=2)
cnts=imutils.grab_contours(cnts)
for c in cnts:
    if cv2.contourArea(c)<area:
        continue
   
        (x,y,w,h)=cv2.boundingRect(c)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    text="moving object detected"
print(text)
cv2.putText(img,text,(10,20),
            cv2.FRONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
cv2.imshow("camerafeed",img)

key=cv2.waitKey(10)
print(key)


