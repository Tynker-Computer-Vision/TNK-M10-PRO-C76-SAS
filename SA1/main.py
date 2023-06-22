from cvzone.FaceDetectionModule import FaceDetector
import cv2
import numpy as np

# Capture the web cam video feed
cap = cv2.VideoCapture(0)

# Create a detector using FaceDetector
detector = FaceDetector()

while True:
    try:
        # Read single image from the captured feed and store it in img variable
        success, img = cap.read()
        # Flip the imgage
        img = cv2.flip(img, 1)
        # Use detector to find faces in the image and store it in bboxs variable
        img, bboxs = detector.findFaces(img, draw=False)

        # Check if bboxs exits
        if bboxs: 
            # Traverse throught each box in the bboxs
            for box in bboxs:
                # Get X, Y, W, H of the box
                X, Y, W, H = box['bbox']

                # Draw the rectangle using X, Y, W, H to show rectangle on detected face
                img = cv2.rectangle(img, (X, Y), (X+W, Y+H), (255, 255, 255), 1)
            
        # Display the image    
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print("Exception", e)

cv2.destroyAllWindows()
