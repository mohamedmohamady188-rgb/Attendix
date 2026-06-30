import cv2
import numpy as np
import json
from insightface.app import FaceAnalysis

class StudentEnroller:
    def __init__(self):

        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        

        self.app.prepare(ctx_id=-1)
        print("Face Enrollment Model Ready (Running on CPU)")

    def extract_encoding(self, image_bytes):
        try:

            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return {"success": False, "message": "Invalid image format."}


            faces = self.app.get(frame)
            
            if len(faces) == 0:
                print("⚠No face detected in the frame")
                return {"success": False, "message": "No face detected. Please ensure good lighting and look directly at the camera."}
            
            if len(faces) > 1:
                print("Multiple faces detected")
                return {"success": False, "message": "Multiple faces detected. Please make sure only the student is in the frame."}


            embedding = faces[0].embedding.astype(float).tolist()
            
            print(f"Face Encoding Extracted Successfully!")
            return {"success": True, "encoding": json.dumps(embedding)}

        except Exception as e:
            print(f"Error during encoding extraction: {str(e)}")
            return {"success": False, "message": f"AI Engine Error: {str(e)}"}

# تجربة سريعة عند تشغيل الملف منفصلاً
if __name__ == "__main__":
    print("Initializing Student Enroller...")
    test_enroller = StudentEnroller()
    print("Ready!")