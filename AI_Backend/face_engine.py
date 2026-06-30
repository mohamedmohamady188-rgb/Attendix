import os
import sys

try:
    import nvidia.cuda_runtime as cuda_runtime
    import nvidia.cublas as cublas
    import nvidia.cufft as cufft
    import nvidia.cudnn as cudnn

    nvidia_modules = [cuda_runtime, cublas, cufft, cudnn]
    if hasattr(os, "add_dll_directory"):
        for module in nvidia_modules:
            if hasattr(module, '__path__'):
                for root, dirs, files in os.walk(module.__path__[0]):
                    if any(f.lower().endswith('.dll') for f in files):
                        os.add_dll_directory(root)
                        os.environ['PATH'] = root + ";" + os.environ["PATH"]
    print("Nvidia libraries is Done")
except Exception as e:
    print(f"Warning libraries installation{e}")

import cv2
import threading
import sqlite3
import numpy as np
import requests
import json
import time
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
from datetime import datetime

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|max_delay;0|analyzeduration;0"


class FaceRecognitionEngine:
    def __init__(self, subject="General", lecture_id=1):
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print('FaceAnalysis Model Ready')

        self.subject = subject
        self.lecture_id = lecture_id
        self.cap = None
        self.running = False
        self.is_scanning = False
        self.current_frame = None
        self.latest_frame = None

        self.cam_thread = None
        self.ai_thread = None

        self.registered_student_ids = set()
        self.frame_count = {}
        self.known_embeddings = []

        self.dotnet_base_url = "http://localhost:5000"
        self.dotnet_record_url = f"{self.dotnet_base_url}/api/Attendance/RecordAttendance"

        try:
            conn = sqlite3.connect('students.db')
            conn.execute('''CREATE TABLE IF NOT EXISTS attendance (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            student_id TEXT,
                            student_name TEXT,
                            subject TEXT,
                            date TEXT,
                            check_in TEXT)''')

            conn.execute('''CREATE TABLE IF NOT EXISTS students (
                            student_id TEXT PRIMARY KEY,
                            full_name TEXT,
                            department TEXT,
                            academic_year TEXT,
                            email TEXT,
                            face_embedding TEXT)''')
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"DB Init Error: {e}")

    def load_known_faces(self):
        print("Fetching faces from Local SQLite Database...")
        self.known_embeddings = []
        try:
            conn = sqlite3.connect('students.db')
            cursor = conn.cursor()
            cursor.execute("SELECT student_id, full_name, face_embedding FROM students")
            rows = cursor.fetchall()

            for row in rows:
                s_id = str(row[0])
                s_name = str(row[1])
                encoding_str = row[2]

                if encoding_str:
                    try:
                        clean_str = str(encoding_str).replace("'", '"')
                        embedding_list = json.loads(clean_str)
                        s_emb = np.array(embedding_list, dtype=np.float32).flatten()
                        self.known_embeddings.append((s_id, s_name, s_emb))
                    except Exception as parse_err:
                        print(f"Failed to parse encoding for {s_name}: {parse_err}")
                        pass

            conn.close()
            print(f"Loaded {len(self.known_embeddings)} faces from Local DB.")
        except Exception as e:
            print(f"Error loading from local DB: {e}")

    # 🛠️ [التعديل الأول]: أضفنا lecture_id كـ Parameter هنا لتحديث رقم الجلسة الحقيقي القادم من الـ .NET عبر الفرونت
    def start_camera(self, course_name="General", lecture_id=1,
                     cam_source="rtsp://admin:asd1234@192.168.1.2:554/user=admin&password=asd1234&channel=0&stream=1.sdp"):
        if self.running: return
        self.subject = course_name
        self.lecture_id = lecture_id  # 🎯 هنا يتم استقبال وتحديث الـ lecture_id الديناميكي بنجاح

        self.load_known_faces()
        self.registered_student_ids = set()
        self.frame_count = {}
        self.is_scanning = False

        if isinstance(cam_source, str):
            print(f"Connecting to DVR Stream: {cam_source}")
            self.cap = cv2.VideoCapture(cam_source, cv2.CAP_FFMPEG)
        else:
            print(f"Connecting to Local Camera Index: {cam_source}")
            self.cap = cv2.VideoCapture(cam_source)

        if not self.cap.isOpened():
            print(f"Could not open camera source: {cam_source}")
            if isinstance(cam_source, str) or cam_source != 0:
                print("Trying internal camera (0) as backup...")
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    print("No camera found at all.")
                    return
            else:
                return

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = True

        self.cam_thread = threading.Thread(target=self._camera_loop)
        self.cam_thread.daemon = True
        self.cam_thread.start()

        self.ai_thread = threading.Thread(target=self._ai_loop)
        self.ai_thread.daemon = True
        self.ai_thread.start()

        print(f"Camera and AI Threads Started for {self.subject} (Lecture ID: {self.lecture_id})")

    def enable_scanning(self):
        self.is_scanning = True
        print("AI Scanning Enabled!")

    def _camera_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.latest_frame = cv2.resize(frame, (640, 480))
            else:
                time.sleep(0.001)

    def _ai_loop(self):
        while self.running:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()

                if not self.is_scanning:
                    self.current_frame = frame
                    time.sleep(0.001)
                    continue

                start_time = time.time()
                faces = self.app.get(frame)

                for face in faces:
                    embedding = np.array(face.embedding, dtype=np.float32).flatten()
                    identity = None

                    for s_id, s_name, s_emb in self.known_embeddings:
                        if cosine(embedding, s_emb) < 0.65:
                            identity = (s_id, s_name)
                            break

                    bbox = face.bbox.astype(int)
                    if identity:
                        s_id, s_name = identity
                        if s_id not in self.registered_student_ids:
                            color = (0, 0, 255)
                            status_text = "Verifying..."
                            self.frame_count[s_id] = self.frame_count.get(s_id, 0) + 1
                            if self.frame_count[s_id] >= 5:
                                proc_time_ms = (time.time() - start_time) * 1000
                                self.save_attendance(s_id, s_name, proc_time_ms)
                                self.registered_student_ids.add(s_id)
                        else:
                            color = (0, 255, 0)
                            status_text = f"Verified: {s_name}"

                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        cv2.putText(frame, status_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,
                                    2)
                    else:
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 165, 255), 2)
                        cv2.putText(frame, "Unknown", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 165, 255), 2)

                total_proc_time = (time.time() - start_time) * 1000
                cv2.putText(frame, f"AI Speed: {total_proc_time:.1f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2)

                self.current_frame = frame
            else:
                time.sleep(0.001)

    def generate_frames(self):
        while self.running:
            if self.current_frame is not None:
                ret, buffer = cv2.imencode('.jpg', self.current_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.01)

    def save_attendance(self, s_id, s_name, proc_time_ms=0.0):
        try:
            conn = sqlite3.connect('students.db')
            cursor = conn.cursor()
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")

            cursor.execute('SELECT id FROM attendance WHERE student_id = ? AND subject = ? AND date = ?',
                           (str(s_id), self.subject, date_str))
            existing_record = cursor.fetchone()

            if not existing_record:
                cursor.execute(
                    'INSERT INTO attendance(student_id, student_name, subject, date, check_in) VALUES (?, ?, ?, ?, ?)',
                    (str(s_id), str(s_name), self.subject, date_str, time_str))
                conn.commit()
                print(f"Saved locally: {s_name} | Process Time: {proc_time_ms:.2f} ms")

                # 🛠️ [التعديل الثاني]: الـ Payload هيسحب تلقائياً الـ self.lecture_id المحدث والديناميكي
                payload = {
                    "studentId": int(s_id),
                    "lectureId": int(self.lecture_id)
                }

                def send_to_dotnet():
                    try:
                        print(f"Sending to .NET: {self.dotnet_record_url} with payload {payload}")
                        res = requests.post(self.dotnet_record_url, json=payload, timeout=10)
                        if res.status_code in [200, 201]:
                            print(f"🎯 .NET SAVED SUCCESSFULLY! Status: {res.status_code}")
                        else:
                            print(f"❌ .NET REJECTED DATA! Status: {res.status_code}")
                            print(f"ERROR MESSAGE: {res.text}")
                    except Exception as e:
                        print(".NET Network Error:", e)

                threading.Thread(target=send_to_dotnet).start()
            else:
                print(
                    f"تنبيه: {s_name} مسجل حضور بالفعل اليوم في هذه المادة! | وقت معالجة الـ AI كان: {proc_time_ms:.1f} ms")

            conn.close()
        except Exception as e:
            print(f"DB Save Error: {e}")

    def get_attendance_records(self):
        try:
            conn = sqlite3.connect('students.db')
            cursor = conn.cursor()
            now = datetime.now().strftime("%Y-%m-%d")
            cursor.execute(
                'SELECT student_id, student_name, check_in FROM attendance WHERE subject = ? AND date = ? ORDER BY id DESC',
                (self.subject, now))
            rows = cursor.fetchall()
            conn.close()

            records = []
            for row in rows:
                db_s_id = row[0]
                db_s_name = row[1]

                for mem_id, mem_name, _ in self.known_embeddings:
                    if mem_id == db_s_id:
                        db_s_name = mem_name
                        break

                records.append({"id": db_s_id, "name": db_s_name, "time": row[2]})

            return records
        except Exception as e:
            return []

    def stop(self):
        self.running = False
        self.is_scanning = False

        if self.cam_thread:
            self.cam_thread.join(timeout=0.5)
        if self.ai_thread:
            self.ai_thread.join(timeout=0.5)

        if self.cap:
            self.cap.release()

        self.current_frame = None
        self.latest_frame = None
        print("Camera and AI Threads Stopped Safely.")