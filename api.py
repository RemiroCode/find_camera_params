from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse  
import subprocess
import os

app = FastAPI()

BASE_LOCATIONS_PATH = "C:/Users/dario/work/blender_img_from_coordinates/"

@app.post("/process/{location_name}")
def process_location(location_name: str):
    script_path = os.path.join(os.getcwd(), "bruteForce.py")
    try:
        # 1. Ejecuta el script que genera el render y guarda la homografía
        subprocess.run(["python", script_path, location_name], check=True)
        
        # 2. Apuntamos al archivo final que generó homograpy.py
        rendered_image_path = os.path.join(BASE_LOCATIONS_PATH, location_name, f"{location_name}_mask_warped.png")
        
        # 3. Si el archivo existe, lo enviamos de vuelta como respuesta binaria
        if os.path.exists(rendered_image_path):
            return FileResponse(rendered_image_path, media_type="image/png")
        else:
            raise HTTPException(status_code=404, detail="El script terminó pero no se encontró la imagen resultante.")
            
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error en bruteForce: {str(e)}")