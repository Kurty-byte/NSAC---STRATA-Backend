from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import PrescribeRequest, PrescribeResponse
from app.services.prescribe import PrescribeEngine

app = FastAPI(title=settings.app_name, debug=settings.debug)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = PrescribeEngine(dataset_path=settings.dataset_path)


@app.get("/health")
def health_check():
    return {"status": "ok", "app": settings.app_name}


@app.post("/api/v1/prescribe", response_model=PrescribeResponse)
def prescribe(req: PrescribeRequest):
    try:
        return engine.prescribe(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
