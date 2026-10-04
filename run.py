import uvicorn

if __name__ == "__main__":
    uvicorn.run("parkpulse.api:app", host="0.0.0.0", port=int(__import__("os").getenv("PORT", "8000")))
