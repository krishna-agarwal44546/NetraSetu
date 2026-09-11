const express = require("express");
const cors = require("cors");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

const app = express();
const PORT = 5000;

// ============================================================
// MIDDLEWARE
// ============================================================

app.use(cors());
app.use(express.json());

// ============================================================
// DIRECTORIES
// ============================================================

const backendDir = __dirname;

// ../ai
const aiDir = path.join(backendDir, "..", "ai");

// Python inference script
const pythonScript = path.join(aiDir, "inference.py");

// Upload directory
const uploadDir = path.join(backendDir, "uploads");

// AI output directory
const outputDir = path.join(aiDir, "outputs");

// Create directories if they don't exist
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}

if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

// ============================================================
// SERVE GRAD-CAM / AI OUTPUTS
// ============================================================

app.use("/outputs", express.static(outputDir));

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

app.post("/api/analyze", upload.single("image"), (req, res) => {

    console.log("\n======================================");
    console.log("New image analysis request");
    console.log("======================================");

    // Check image
    if (!req.file) {

        return res.status(400).json({
            status: "error",
            message: "No image uploaded"
        });
    }

    const imagePath = req.file.path;

    console.log("Uploaded image:");
    console.log(imagePath);

    console.log("\nPython script:");
    console.log(pythonScript);

    // Check inference.py
    if (!fs.existsSync(pythonScript)) {

        console.error("ERROR: inference.py not found");

        return res.status(500).json({
            status: "error",
            message: "Python inference script not found",
            path: pythonScript
        });
    }

    console.log("\nStarting PyTorch inference...");

    // ========================================================
    // RUN PYTHON
    // ========================================================

    const python = spawn("python3", [
        pythonScript,
        imagePath
    ]);

    let stdout = "";
    let stderr = "";

    // ========================================================
    // PYTHON STDOUT
    // ========================================================

    python.stdout.on("data", (data) => {

        const text = data.toString();

        stdout += text;

        console.log("[AI OUTPUT]", text.trim());
    });

    // ========================================================
    // PYTHON STDERR
    // ========================================================

    python.stderr.on("data", (data) => {

        const text = data.toString();

        stderr += text;

        console.error("[AI LOG]", text.trim());
    });

    // ========================================================
    // PYTHON ERROR
    // ========================================================

    python.on("error", (error) => {

        console.error("Failed to start Python:");
        console.error(error);

        return res.status(500).json({
            status: "error",
            message: "Could not start Python",
            details: error.message
        });
    });

    // ========================================================
    // PYTHON FINISHED
    // ========================================================

    python.on("close", (code) => {

        console.log("\nPython process finished");
        console.log("Exit code:", code);

        // ----------------------------------------------------
        // PYTHON FAILED
        // ----------------------------------------------------

        if (code !== 0) {

            console.error("Python inference failed");

            console.error(stderr);

            return res.status(500).json({
                status: "error",
                message: "AI inference failed",
                details: stderr
            });
        }

        // ----------------------------------------------------
        // PYTHON SUCCESS
        // ----------------------------------------------------

        try {

            /*
             * inference.py should print ONLY JSON to stdout.
             *
             * Example:
             *
             * {
             *   "status": "success",
             *   "prediction": "Moderate DR",
             *   "confidence": 0.91,
             *   "quality": {...},
             *   "gradcam": "fundus_gradcam.jpg"
             * }
             */

            const result = JSON.parse(stdout.trim());

            // ------------------------------------------------
            // ADD GRAD-CAM URL
            // ------------------------------------------------

            if (result.gradcam) {

                // If Python returns:
                // fundus_gradcam.jpg

                result.gradcamUrl =
                    `http://localhost:${PORT}/outputs/${path.basename(result.gradcam)}`;
            }

            // ------------------------------------------------
            // ADD ORIGINAL IMAGE URL
            // ------------------------------------------------

            result.imageUrl =
                `http://localhost:${PORT}/uploads/${path.basename(imagePath)}`;

            console.log("\nAI RESULT:");
            console.log(result);

            // ------------------------------------------------
            // SEND TO REACT
            // ------------------------------------------------

            return res.json(result);

        } catch (error) {

            console.error("\nJSON parsing failed!");

            console.error("Raw Python output:");
            console.error(stdout);

            return res.status(500).json({
                status: "error",
                message: "Python returned invalid JSON",
                pythonOutput: stdout,
                pythonError: stderr
            });
        }
    });
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

    console.log(`Server: http://localhost:${PORT}`);

    console.log("\nPython:");
    console.log(pythonScript);

    console.log("\nUpload directory:");
    console.log(uploadDir);

    console.log("\nAI output directory:");
    console.log(outputDir);

    console.log("\nWaiting for images...\n");
});