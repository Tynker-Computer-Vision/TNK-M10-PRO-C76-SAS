from cvzone.FaceDetectionModule import FaceDetector
import cv2
import numpy as np
from keras.models import load_model

cap = cv2.VideoCapture(0)

detector = FaceDetector()

ageDetectionModel = load_model('age_model_50epochs.h5', compile=False)

while True:
    try:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        
        img, bboxs = detector.findFaces(img, draw=False)

        if bboxs: 
            for box in bboxs:
                X, Y, W, H = box['bbox']

                croppedImg = img[Y:Y+H, X:X+W]
                resizedImg = cv2.resize(croppedImg, (200, 200))
                resizedImg = np.array([resizedImg])
                
                prediction = ageDetectionModel.predict(resizedImg)

                # Create a color variable and give it white color 
                color = (255, 255, 255)

                # Check if prediction[0][0] i.e age is less then 20 and assign a color to color variable
                if prediction[0][0] < 80:
                    color = (255, 0, 255)
                # Check if prediction[0][0] i.e age is less then 40 and assign a color to color variable
                if prediction[0][0] < 60:
                    color = (255, 255, 0)
                # Check if prediction[0][0] i.e age is less then 60 and assign a color to color variable
                if prediction[0][0] < 40:
                    color = (0, 255, 255)
                # Check if prediction[0][0] i.e age is less then 80 and assign a color to color variable
                if prediction[0][0] < 20:
                    color = (255, 0, 0)

                # Use color variable to give color to the text
                img = cv2.putText(img, str(int(prediction[0][0])), (X, Y), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
                 
                # Use color variable to give color to the rectangle
                img = cv2.rectangle(img, (X, Y), (X+W, Y+H), color, 1)
             
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print("Exception", e)

cv2.destroyAllWindows()
