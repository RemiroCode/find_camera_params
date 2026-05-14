from fastapi import FastAPI, HTTPException
import subprocess
import os

app = FastAPI()

@app.post("/process/{location_name}")
def process_location(location_name: str):
    script_path = os.path.join(os.getcwd(), "bruteForce.py")
    try:
        subprocess.run(["python", script_path, location_name], check=True)
        return {"status": "success", "location": location_name}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error en bruteForce: {str(e)}")