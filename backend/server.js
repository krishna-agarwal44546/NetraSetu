const express = require("express");
const cors = require("cors");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const axios = require("axios");
const FormData = require("form-data");

const app = express();
const PORT = process.env.PORT || 5000;

// ============================================================
// AI SERVICE URL
// ============================================================

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || "https://netrasetu-1.onrender.com";

// ============================================================
// MIDDLEWARE
// ============================================================

app.use(cors({ origin: process.env.FRONTEND_URL || "https://netrasetu-3.onrender.com" }));
app.use(express.json());

// ============================================================
// DIRECTORIES
// ============================================================

const backendDir = __dirname;

// Upload directory
const uploadDir = path.join(backendDir, "uploads");

// Create directory if it doesn't exist
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}

// ============================================================
// MULTER CONFIGURATION
// ============================================================

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, uploadDir);
    },

    filename: function (req, file, cb) {
        const extension = path.extname(file.originalname);

        const filename =
            "fundus_" +
            Date.now() +
            extension;

        cb(null, filename);
    }
});

const upload = multer({
    storage: storage,

    limits: {
        fileSize: 10 * 1024 * 1024 // 10 MB
    },

    fileFilter: function (req, file, cb) {

        const allowedTypes = [
            "image/jpeg",
            "image/jpg",
            "image/png"
        ];

        if (allowedTypes.includes(file.mimetype)) {
            cb(null, true);
        } else {
            cb(new Error("Only JPG, JPEG and PNG images are allowed"));
        }
    }
});

// ============================================================
// TEST ROUTE
// ============================================================

app.get("/", (req, res) => {
    res.json({
        status: "ok",
        message: "Diabetic Retinopathy AI backend is running"
    });
});

// ============================================================
// AI ANALYSIS ROUTE
// ============================================================

app.post("/api/analyze", upload.single("image"), async (req, res) => {
    console.log("\n======================================");
    console.log("New image analysis request");
    console.log("======================================");

    if (!req.file) {
        return res.status(400).json({ status: "error", message: "No image uploaded" });
    }

    const imagePath = req.file.path;
    console.log("Uploaded image:", imagePath);
    console.log("Forwarding to AI service:", AI_SERVICE_URL);

    try {
        const form = new FormData();
        form.append("image", fs.createReadStream(imagePath));

        const aiResponse = await axios.post(`${AI_SERVICE_URL}/analyze`, form, {
            headers: form.getHeaders(),
            timeout: 60000 // AI inference can be slow on Render's free tier / cold starts
        });

        const result = aiResponse.data;

        const baseUrl = `${req.protocol}://${req.get("host")}`;

        // inference.py nests the heatmap path under explainability.gradcam,
        // not top-level "gradcam" — read it from the right place.
        if (result.explainability && result.explainability.gradcam) {
            result.gradcamUrl = `${AI_SERVICE_URL}/outputs/${path.basename(result.explainability.gradcam)}`;
        }

        result.imageUrl = `${baseUrl}/uploads/${path.basename(imagePath)}`;

        console.log("\nAI RESULT:", result);
        return res.json(result);

    } catch (error) {
        console.error("\nAI service call failed:");
        console.error(error.message);

        return res.status(500).json({
            status: "error",
            message: "AI inference failed",
            details: error.response?.data || error.message
        });
    }
});

// ============================================================
// SERVE UPLOADED IMAGES
// ============================================================

app.use("/uploads", express.static(uploadDir));

// ============================================================
// MULTER / GENERAL ERROR HANDLER
// ============================================================

app.use((err, req, res, next) => {

    console.error("Server error:", err);

    return res.status(500).json({
        status: "error",
        message: err.message || "Server error"
    });
});

// ============================================================
// START SERVER
// ============================================================

app.listen(PORT, () => {

    console.log("\n======================================");
    console.log("Diabetic Retinopathy Backend");
    console.log("======================================");

    console.log(`Server running on port: ${PORT}`);
    console.log("AI service URL:", AI_SERVICE_URL);
    console.log("Upload directory:", uploadDir);

    console.log("\nWaiting for images...\n");
});
