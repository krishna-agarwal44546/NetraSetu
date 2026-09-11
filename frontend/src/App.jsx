import { useState } from "react";
import "./App.css";

const API_URL = "http://localhost:5000";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // =========================================================
  // SELECT IMAGE
  // =========================================================

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];

    if (!selectedFile) return;

    setFile(selectedFile);
    setResult(null);
    setError("");

    // Preview image
    const imageUrl = URL.createObjectURL(selectedFile);
    setPreview(imageUrl);
  };

  // =========================================================
  // ANALYZE IMAGE
  // =========================================================

  const analyzeImage = async () => {
    if (!file) {
      setError("Please select a fundus image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();

      // MUST MATCH multer upload.single("image")
      formData.append("image", file);

      console.log("Sending image to backend...");

      const response = await fetch(`${API_URL}/api/analyze`, {
        method: "POST",
        body: formData,
      });

      console.log("HTTP status:", response.status);

      const data = await response.json();

      console.log("Backend response:", data);

      // Backend returned an error
      if (!response.ok || data.status === "error") {
        throw new Error(
          data.message ||
            data.details ||
            "AI analysis failed."
        );
      }

      // Save result
      setResult(data);

    } catch (err) {
      console.error("Analysis error:", err);

      setError(err.message || "AI analysis failed.");
    } finally {
      setLoading(false);
    }
  };

  // =========================================================
  // RESET
  // =========================================================

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");
  };

  // =========================================================
  // UI
  // =========================================================

  return (
    <div className="app">

      <header>
        <h1>Diabetic Retinopathy Screening</h1>

        <p>
          AI-assisted fundus image analysis
        </p>
      </header>

      {/* ====================================================
          UPLOAD SECTION
      ==================================================== */}

      <div className="card">

        <input
          type="file"
          accept="image/png,image/jpeg,image/jpg"
          onChange={handleFileChange}
        />

        {file && (
          <p className="filename">
            Selected: {file.name}
          </p>
        )}

        {/* Image Preview */}

        {preview && (
          <div className="preview-container">

            <img
              src={preview}
              alt="Fundus preview"
              className="fundus-image"
            />

          </div>
        )}

        {/* Analyze */}

        <button
          className="analyze-button"
          onClick={analyzeImage}
          disabled={!file || loading}
        >
          {loading ? "Analyzing..." : "Analyze Image"}
        </button>

        {file && (
          <button
            className="reset-button"
            onClick={reset}
          >
            Reset
          </button>
        )}

      </div>

      {/* ====================================================
          ERROR
      ==================================================== */}

      {error && (
        <div className="error">
          <strong>Error:</strong>

          <p>{error}</p>
        </div>
      )}

      {/* ====================================================
          RESULT
      ==================================================== */}

      {result && (
        <div className="results">

          <h2>AI Analysis Result</h2>

          {/* ----------------------------------------------
              IMAGE QUALITY
          ---------------------------------------------- */}

          {result.quality && (
            <div className="result-card">

              <h3>Image Quality</h3>

              <div className="quality-status">
                {result.quality.usable ? (
                  <span className="good">
                    ✓ Image usable
                  </span>
                ) : (
                  <span className="bad">
                    ✗ Image not usable
                  </span>
                )}
              </div>

              <p>
                Quality Score:{" "}
                <strong>
                  {result.quality.score}
                </strong>
              </p>

              {result.quality.metrics && (
                <div className="metrics">

                  <p>
                    Blur Score:{" "}
                    {result.quality.metrics.blurScore}
                  </p>

                  <p>
                    Brightness:{" "}
                    {result.quality.metrics.brightness}
                  </p>

                  <p>
                    Contrast:{" "}
                    {result.quality.metrics.contrast}
                  </p>

                  <p>
                    Saturation:{" "}
                    {result.quality.metrics.saturation}
                  </p>

                  <p>
                    Field of View:{" "}
                    {result.quality.metrics.fieldOfViewRatio}
                  </p>

                  <p>
                    Resolution:{" "}
                    {result.quality.metrics.width} ×{" "}
                    {result.quality.metrics.height}
                  </p>

                </div>
              )}

              {result.quality.issues &&
                result.quality.issues.length > 0 && (
                  <div className="issues">

                    <strong>Issues:</strong>

                    <ul>
                      {result.quality.issues.map(
                        (issue, index) => (
                          <li key={index}>
                            {issue}
                          </li>
                        )
                      )}
                    </ul>

                  </div>
                )}

            </div>
          )}

          {/* ----------------------------------------------
              PREDICTION
          ---------------------------------------------- */}

          {result.prediction && (
            <div className="result-card prediction">

              <h3>DR Prediction</h3>

              <div className="prediction-class">
                {result.prediction.class}
              </div>

              <p>
                Grade:{" "}
                <strong>
                  {result.prediction.grade}
                </strong>
              </p>

              <p>
                Severity Score:{" "}
                <strong>
                  {Number(
                    result.prediction.severityScore
                  ).toFixed(3)}
                </strong>
              </p>

            </div>
          )}

          {/* ----------------------------------------------
              EXPLAINABILITY
          ---------------------------------------------- */}

          {result.explainability && (
            <div className="result-card">

              <h3>Explainability — Grad-CAM</h3>

              <p>
                {result.explainability.description}
              </p>

              {result.explainability.gradcam && (
                <img
                  src={
                    result.gradcamUrl ||
                    `${API_URL}/outputs/${result.explainability.gradcam
                      .split("/")
                      .pop()}`
                  }
                  alt="Grad-CAM explanation"
                  className="gradcam-image"
                />
              )}

            </div>
          )}

          {/* ----------------------------------------------
              RECOMMENDATION
          ---------------------------------------------- */}

          {result.recommendation && (
            <div className="recommendation">

              <h3>Recommendation</h3>

              <p>
                {result.recommendation}
              </p>

            </div>
          )}

        </div>
      )}

    </div>
  );
}

export default App;