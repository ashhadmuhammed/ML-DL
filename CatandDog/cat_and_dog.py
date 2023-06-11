import cv2
import numpy as np
import time
import random
from tensorflow.keras.models import load_model
model = load_model(r'cat_and_dog.h5')
cap = cv2.VideoCapture(0)

while True:

    
    
            return_value,frame = cap.read()
            
            frame = cv2.flip(frame,1)

            if not return_value:
                continue

            cv2.rectangle(frame,(380,80),(636,336),(0,255,255),2)
            hand_img = frame[80:336,380:636]



        
        
            img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
            
            x=np.array([img],"float32")
            x/=255
            test_input = x.reshape((1,256,256,3))
            a=model.predict(x)
            print(a[0][0])
            if(a[0][0]>0.8):
                   cv2.putText(frame,"Dog",(150,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            elif(a[0][0]<0.2 and a[0][0]>0.1):
                   cv2.putText(frame,"Cat",(150,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            elif(a[0][0]<0.1):
                    cv2.putText(frame,"Nothing",(150,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Hand Cricket", frame)
            k = cv2.waitKey(10)
            if k == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()


