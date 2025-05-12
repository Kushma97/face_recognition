from deepface import DeepFace
import cv2
import os

db_path = "rcaptured_faces"

video_capture = cv2.VideoCapture(0)

print("Starting camera. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        results = DeepFace.find(img_path=frame_rgb, db_path=db_path, enforce_detection=False)
        if len(results) > 0:
            name = os.path.basename(results[0]['identity'][0]).split('.')[0]
        else:
            name = "Unknown"
    except Exception as e:
        name = "Unknown"

    cv2.putText(frame, name, (350, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 6)
    cv2.putText(frame, "Press 'q' to quit", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    cv2.imshow('Live Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
