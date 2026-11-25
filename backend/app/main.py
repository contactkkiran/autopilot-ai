from fastapi import FastAPI

app = FastAPI(
    title="AutoPilot AI",
    description="Agentic Vehicle Intelligence Platform",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "Welcome to AutoPilot AI! ", "status": "running"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "application": "AutoPilot AI"}
