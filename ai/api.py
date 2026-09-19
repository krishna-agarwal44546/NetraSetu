from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
import tempfile
import os

from inference import analyze

app = FastAPI()

# ============================================================
# SERVE GRAD-CAM OUTPUTS
# ============================================================
# inference.py writes heatmaps to ./outputs (see OUTPUT_DIR).
# Mounting it here lets server.js build a public URL for it.

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUT_DIR),
    name="outputs"
)


@app.get("/health")
def health():
    return {"status": "ok"}


# Renamed from /predict to /analyze, and the field renamed from
# "file" to "image", to match what server.js sends.
@app.post("/analyze")
async def analyze_endpoint(image: UploadFile = File(...)):

    suffix = os.path.splitext(image.filename)[1] or ".jpg"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:
        temp.write(await image.read())
        image_path = temp.name

    try:
        return analyze(image_path)

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
