require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');


const app = express();
app.use(cors());
app.use(express.json());

// --- 1. DATABASE CONNECTION ---
// Replace this string if you are using MongoDB Atlas
const PORT = process.env.PORT || 5000;
const MONGO_URI = process.env.MONGO_URI;

mongoose.connect(MONGO_URI)
    .then(() => console.log("✅ MongoDB Connected"))
    .catch(err => console.error("❌ DB Connection Error:", err));

// --- 2. DATABASE SCHEMA (Patient History) ---
const ReportSchema = new mongoose.Schema({
    patientName: String,
    age: Number,
    sex: String,
    module: String, // 'chest', 'brain', 'eye', 'skin'
    diagnosis: String,
    confidence: Number,
    date: { type: Date, default: Date.now }
});
const Report = mongoose.model('Report', ReportSchema);

// --- 3. FILE UPLOAD CONFIG ---
const upload = multer({ dest: 'uploads/' });

// --- 4. THE MASTER API ROUTE ---
app.post('/api/analyze', upload.single('image'), async (req, res) => {
    try {
        const { module, patientName, age, sex, site } = req.body;
        const filePath = req.file.path;

        console.log(`received request for module: ${module}`);

        // A. PREPARE DATA FOR PYTHON API
        const formData = new FormData();
        formData.append('file', fs.createReadStream(filePath));
        
        // Add extra fields if it's Skin Cancer
        if (module === 'skin') {
            formData.append('age', age);
            formData.append('sex', sex);
            formData.append('site', site);
        }

        // B. CALL PYTHON API (The "Bridge")
        // Note: Python is running on port 8000
        const pythonUrl = `http://127.0.0.1:8000/predict/${module}`;
        const response = await axios.post(pythonUrl, formData, {
            headers: { ...formData.getHeaders() }
        });

        const aiResult = response.data;

        // C. SAVE TO DATABASE (History)
        let diagnosis_text = aiResult.diagnosis || aiResult.findings?.[0]?.condition || "Unknown";
        let confidence_score = aiResult.confidence || aiResult.findings?.[0]?.confidence || 0;

        const newReport = new Report({
            patientName: patientName || "Anonymous",
            age: age || 0,
            sex: sex || "Unknown",
            module: module,
            diagnosis: diagnosis_text,
            confidence: confidence_score
        });

        await newReport.save();
        console.log("✅ Report saved to Database");

        // D. CLEANUP & RESPOND
        fs.unlinkSync(filePath); // Delete temp file
        res.json({ success: true, data: aiResult, historyId: newReport._id });

    } catch (error) {
        console.error("Error:", error.message);
        res.status(500).json({ success: false, error: "AI Processing Failed" });
    }
});

// --- 5. HISTORY ROUTE ---
app.get('/api/history', async (req, res) => {
    try {
        const reports = await Report.find().sort({ date: -1 });
        res.json(reports);
    } catch (err) {
        res.status(500).json({ error: "Failed to fetch history" });
    }
});

app.listen(PORT, () => console.log(`🚀 Node Server running on port ${PORT}`));