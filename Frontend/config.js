// config.js - Localhost Configuration for Live Demo

const CONFIG = {
    // 🐍 رابط سيرفر بايثون
    PYTHON_API_URL: "http://127.0.0.1:8000",

    // 🌐 لينك الباك إند (.NET) بعد تصليح البورت
    DOTNET_API_URL: "http://localhost:5000/api",

    // 🗄 قاعدة بيانات وهمية (Mock Database) للتشغيل المحلي السريع
    MOCK_DATABASE: {
        users: [
            { email: "yousseftarek163@sha.edu.eg", password: "123", role: "admin", name: "Youssef Tarek" },
            { email: "doctor@sha.edu.eg", password: "123", role: "doctor", name: "Dr. Ahmed" }
        ],
        initialStudents: [
            { id: "423240067", name: "Mostafa Taha", email: "423240067@sha.edu.eg", department: "CS" },
            { id: "423240069", name: "Yousef Tarek", email: "423240069@sha.edu.eg", department: "IS" }
        ]
    }
};

window.API_CONFIG = CONFIG;