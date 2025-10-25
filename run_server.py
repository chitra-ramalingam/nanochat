import uvicorn
if __name__ == "__main__":
    uvicorn.run("scripts.chat_web:app", host="127.0.0.1", port=8000, reload=False)
