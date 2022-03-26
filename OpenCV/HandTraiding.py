from curses import COLOR_YELLOW
import cv2
import mediapipe as mp
import numpy as np

face_dec = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)


with face_dec.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    while True:


        
    
        succes, img = cap.read()

        img.flags.writeable = False

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = face_detection.process(img)


        img.flags.writeable = True

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if results.detections:
            for detection in results.detections:
                
                if(len(results.detections) > 1):
                    img = cv2.putText(img, "Remove your face", (int(np.array(img).shape[0]/2), int(np.array(img).shape[1]/2)), cv2.FONT_HERSHEY_SIMPLEX ,1, COLOR_YELLOW, 6)
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    
                mp_drawing.draw_detection(img, detection)
                print(detection)

        
        # img = cv2.putText(img, "Remove your face", (int(np.array(img).shape[0]/2), int(np.array(img).shape[1]/2)), cv2.FONT_HERSHEY_SIMPLEX ,1, COLOR_YELLOW, 6)
        cv2.imshow('MediaPipe Face Detection', img)

        # if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) <= 1:
        #     break
        if cv2.waitKey(5) & 0xFF == 27:
            break
        
        cv2.waitKey(1)

