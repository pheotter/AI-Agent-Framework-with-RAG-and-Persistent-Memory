from fastapi import FastAPI
from app.routes import chat

app = FastAPI(
    title="AI Agent API",
    description="Chat-based AI agent with retrieval-augmented generation and persistent memory",
    version="1.0.0"
)

# Include routers
app.include_router(chat.router)

@app.get("/")
async def root():
    """Root endpoint that returns a welcome message"""
    return {"message": "Welcome to the AI Agent API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
