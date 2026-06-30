import sqlite3

def init_db():
    try:
        conn = sqlite3.connect('students.db')
        cursor = conn.cursor()
        
        # جدول تسجيل الغياب (عشان الداشبورد تقرأ منه لايف)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                student_name TEXT,
                subject TEXT,
                date TEXT,
                check_in TEXT
            )
        ''')
        
        # جدول احتياطي للطلاب (لو النت فصل والـ .NET وقع)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                full_name TEXT,
                department TEXT,
                academic_year TEXT,
                email TEXT,
                face_embedding BLOB
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Local SQLite Database Initialized Successfully.")
    except Exception as e:
        print(f"❌ Database Error: {e}")

if __name__ == "__main__":
    init_db()