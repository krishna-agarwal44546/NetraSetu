from fastapi import FastAPI, UploadFile, File
import tempfile
import os

from inference import analyze

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    suffix = os.path.splitext(file.filename)[1] or ".jpg"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:
        temp.write(await file.read())
        image_path = temp.name

    try:
        return analyze(image_path)

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)