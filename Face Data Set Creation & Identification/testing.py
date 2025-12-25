import cv2,numpy,os
haar_file="C:\\Users\\Anbazhagan R\\Desktop\\Face Data Set Creation & Identification\\haarcascade_frontalface_default.xml"
datasets="C:\\Users\\Anbazhagan R\\Desktop\\Face Data Set Creation & Identification\data set"
print('training....')
(image,labels,names,id)=([],[],{},0)
for (subdirs,dirs,files)in os.walk(datasets):
    for subdir in dirs:
        subjectpath = os.path.join(datasets,subdir)
        for filename in os.listdir(subjectpath):
            path=subjectpath+'/'+filename
            label=id 
            image.append(cv2.imread(path,0))
            labels.append(int(label))
            #print(lables)
            id+= 1
(width,height)=(130,100)
(image,labels)=[numpy.array(lis) for lis in [image,labels]]

#PRINT(images,labels)
#model =cv2.face.lbphf face face regonizer create ()
model = cv2.face.FisherFaceRecognizer_create()
model.train(image,labels)

face_cascade=cv2.CascadeClassifier(haar_file)
webcam=cv2.VideoCapture(0)
cnt=0
while True:
    (_,im)=webcam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,4)
    
    for (x,y,w,h) in faces:
        face=gray[y:y+h,x:x+w]
        face_resize=cv2.resize(face,(width,height))
        
        
        predection=model.predict(face_resize)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
        if predection[1]<800:
            cv2.putText(im,'%s-%.0f'%(names[predection[0]],predection[1],(x-10,y-10),cv2.FONT__HERSHEY_COMPLEX,1,(51,255,255)))
            print(names[predection[0]])
            cnt=0

        else:
            cnt+1==1
            cv2.putText(im,'unknown',(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
            
            if(cnt>100):
                print("unknown person ")
                cv2.imwrite("input.jpg",im)
                cnt=0
    cv2.imshow('opencv',im)
    key=cv2.waitKey(100)
    if key==27:
        break
webcam.release()
cv2.destroyAllWindows()

    
        