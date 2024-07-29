from fastapi import FastAPI
from app.api.routers import router as api_router

app = FastAPI()

# Include the router for API routes
app.include_router(api_router, prefix="/api")
