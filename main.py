import os
import tempfile
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from predictor import SimilarityPredictor  # <-- move your model code to predictor.py

# Load model once at startup
MODEL_PATH = os.getenv("MODEL_PATH", "best_model.pth")
THRESHOLD = float(os.getenv("THRESHOLD", 0.5))
predictor = SimilarityPredictor(MODEL_PATH, threshold=THRESHOLD)

app = FastAPI()

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    text: str = Form(...)
):
    try:
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        # Run prediction
        result = predictor.predict_similarity(tmp_path, text, verbose=False)

        # Cleanup temp file
        os.remove(tmp_path)

        if result is None:
            return JSONResponse({"error": "Prediction failed"}, status_code=500)

        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
def home():
    return {"message": "Image-Text Similarity API is running"}
