from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sqlite3
import json
import csv
import io

from add_student import StudentEnroller
from face_engine import FaceRecognitionEngine
import database

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

database.init_db()

enroller = StudentEnroller()
engine = FaceRecognitionEngine()

if not hasattr(engine, 'verification_speeds'):
    engine.verification_speeds = {}

original_save_attendance = engine.save_attendance

def patched_save_attendance(s_id, s_name, proc_time_ms=0.0):
    engine.verification_speeds[str(s_id)] = f"{proc_time_ms:.1f} ms"
    return original_save_attendance(s_id, s_name, proc_time_ms)

engine.save_attendance = patched_save_attendance

@app.post("/api/get-face-encoding")
async def get_face_encoding(
        file: UploadFile = File(...),
        std_id: str = Form(None),
        std_name: str = Form(None),
        std_dept: str = Form(None),
        std_year: str = Form(None),
        std_email: str = Form(None)
):
    try:
        contents = await file.read()
        result = enroller.extract_encoding(contents)

        if result["success"]:
            if std_id and std_name:
                conn = sqlite3.connect('students.db')
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO students (student_id, full_name, department, academic_year, email, face_embedding)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (std_id, std_name, std_dept, std_year, std_email, result["encoding"]))
                conn.commit()
                conn.close()
                engine.load_known_faces()

            return {
                "success": True,
                "encoding": result["encoding"],
                "message": "AI Processing Complete"
            }
        else:
            return JSONResponse(status_code=400, content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})


@app.get("/start_camera")
def start_camera(course_name: str = "General", lecture_id: str = "1"):
    # بنحاول نحول الـ ID لرقم عشان يروح للـ .NET مظبوط
    try:
        clean_lecture_id = int(lecture_id)
    except ValueError:
        print(f"⚠️ تنبيه: الفرونت إند باعت الـ ID نصي '{lecture_id}'. تم تحويله إلى 1 مؤقتاً لتوافق الـ .NET")
        clean_lecture_id = 1

    engine.start_camera(course_name=course_name, lecture_id=clean_lecture_id)
    return {"status": "Camera Initialized"}

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(engine.generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/start_model")
def start_model():
    engine.enable_scanning()
    return {"status": "AI Scanning Started"}


@app.get("/stop_model")
def stop_model():
    engine.stop()
    return {"status": "System Stopped"}


@app.get("/get_live_attendance")
def get_live_attendance():
    records = engine.get_attendance_records()
    for r in records:
        r['speed'] = engine.verification_speeds.get(str(r['id']), "0.0 ms")
    return {"attendance": records}


@app.get("/export_report")
def export_report(course_name: str = "General"):
    conn = sqlite3.connect('students.db')
    cursor = conn.cursor()
    cursor.execute('SELECT student_id, student_name, date, check_in FROM attendance WHERE subject=?', (course_name,))
    rows = cursor.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Student ID', 'Student Name', 'Date', 'Time'])
    writer.writerows(rows)
    output.seek(0)

    filename = f"Attendance_{course_name.replace(' ', '_')}.csv"
    return StreamingResponse(output, media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename={filename}"})