import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import subprocess
import os

app = FastAPI(title="Vision Doctor Launcher")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
# 웹사이트 서버 실행용
# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/launch/farm")
async def launch_farm():
    try:
        # Run vision_doctor_system.py asynchronously
        # Use python executable from active environment
        import sys
        python_exe = sys.executable
        subprocess.Popen([python_exe, "vision_doctor_system.py"], 
                         cwd=os.path.dirname(os.path.abspath(__file__)))
        return {"status": "success", "message": "Vision Doctor (Farm Scan) launched."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/launch/cctv")
async def launch_cctv():
    try:
        import sys
        python_exe = sys.executable
        subprocess.Popen([python_exe, "webcam_inference.py"], 
                         cwd=os.path.dirname(os.path.abspath(__file__)))
        return {"status": "success", "message": "Live CCTV Monitor launched."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    uvicorn.run("launcher:app", host="0.0.0.0", port=8000, reload=True)
