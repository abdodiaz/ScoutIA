from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="ScoutIA Pro - API")

@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})

@app.get("/predict")
def predict(dummy: bool = False):
    # TODO: integrate model inference
    return {"prediction": "not_implemented", "dummy": dummy}
