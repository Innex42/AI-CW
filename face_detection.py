import cv2
import pathlib
from keras.models import load_model
from time import sleep
from keras.utils.image_utils import img_to_array
#from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np

cascade_path = "./data/haarcascade_frontalface_default.xml"
emotion_model = load_model('./emotion_detection_model_50epochs.h5')
age_model = load_model('./age_model_50epochs.h5')
gender_model = load_model('./gender_model_50epochs.h5')
#print(cascade_path)

clf = cv2.CascadeClassifier(str(cascade_path))

emotion_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']
gender_labels = ['Male', 'Female']


# Videocapture(0) is your Pc's deafult video input (so if a webcam is connected it should use that).
camera = cv2.VideoCapture(0)
#camera= cv2.imread("abba.png")

while True:
    _, frame = camera.read()
    labels=[]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 255, 0), 2)
        roi_gray=gray[y:y+height,x:x+width]
        roi_gray=cv2.resize(roi_gray,(48,48), interpolation=cv2.INTER_AREA)

        #Get Image ready for prediction
        roi=roi_gray.astype('float')/255.0 #Scale
        roi=img_to_array(roi)
        roi=np.expand_dims(roi,axis=0) #expand dimensions to get it ready for prediction (1, 48, 48, 1)

        emotion_prediction=emotion_model.predict(roi)[0] # Yeild one hot encoded result for 7 classes
        label=emotion_labels[emotion_prediction.argmax()] # find the label
        label_position=(x,y)
        cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

     #Gender
        roi_color=frame[y:y+height,x:x+width]
        roi_color=cv2.resize(roi_color,(200,200),interpolation=cv2.INTER_AREA)
        gender_predict = gender_model.predict(np.array(roi_color).reshape(-1,200,200,3))
        gender_predict = (gender_predict>= 0.5).astype(int)[:,0]
        gender_label=gender_labels[gender_predict[0]] 
        gender_label_position=(x,y+height+50) #50 pixels below to move the label outside the face
        cv2.putText(frame,gender_label,gender_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        #Age
        age_predict = age_model.predict(np.array(roi_color).reshape(-1,200,200,3))
        age = round(age_predict[0,0])
        age_label_position=(x+height,y+height)
        cv2.putText(frame,"Age="+str(age),age_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    

    cv2.imshow("Emotion, Age and Gender Detector", frame)
    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows