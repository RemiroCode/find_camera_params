
from RDD.RDD import build
from RDD.RDD_helper import RDD_helper
from matplotlib import pyplot as plt
from time import time
import sys
RDD_model = build(weights='./weights/RDD-v2.pth')
RDD_model.eval()
RDD = RDD_helper(RDD_model)

import bpy
import os
import numpy as np
from PIL import Image
#import cma
import math
import json
import cv2

import re
import cv2
import pandas as pd


from matplotlib import pyplot as plt

from mathutils import Vector

# ---------- CONFIG ----------
import sys
import os

# Nombre del directorio que nos pasa la API
LOCATION = sys.argv[1] if len(sys.argv) > 1 else "fuego14"
GLOBAL_PATH = "cma_optim"

# Leemos el volumen de Docker, si no existe usamos la carpeta local
LOCATIONS_PATH = os.environ.get("LOCATIONS_PATH", os.path.join(os.getcwd(), "locations"))

ORTO_PATH =  os.path.join(LOCATIONS_PATH, LOCATION, "orto.png")
MDT_PATH =  os.path.join(LOCATIONS_PATH, LOCATION, "mdt.tif")
OUTPUT_FOLDER =  os.path.join(LOCATIONS_PATH, LOCATION)
TARGET_IMAGE_PATH =  os.path.join(LOCATIONS_PATH, LOCATION, f"{LOCATION}.jpeg")

# Camera name in the Blender scene
CAMERA_NAME = "Camera"

# Image resolution (must match target)
TARGET_IMG = cv2.imread(TARGET_IMAGE_PATH)

if TARGET_IMG is None:
    raise ValueError(f"No se pudo cargar la imagen objetivo en {TARGET_IMAGE_PATH}")

RES_Y, RES_X = TARGET_IMG.shape[0], TARGET_IMG.shape[1] 

RENDER_OUTPUT_PATH = os.path.join(LOCATIONS_PATH, LOCATION, "cma_optim", "iterations")

INFINITY = 1e10
DEBUG = False
# ----------------------------

def load_target_image(path):
    img = Image.open(path).convert("RGB").resize((RES_X, RES_Y))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

#TARGET_IMG = load_target_image(TARGET_IMAGE_PATH)

def mkfolder(m_path):
    if  not os.path.isdir(m_path):
        os.mkdir(m_path)


def set_camera_from_params(params):
    """
    params: [tx, ty, tz, rx, ry, rz]
    Positions in Blender units, rotations in radians (XYZ Euler).
    """
    # cam = bpy.data.objects[CAMERA_NAME]
    cam = bpy.data.objects.get(CAMERA_NAME)
    if len(params) == 5:    
        tz, rx, ry, rz, fov = params
        tx = 0
        ty = 0
    elif len(params) == 7:
        tx, ty, tz, rx, ry, rz, fov = params
    else:
        print("Number of parameters not supported. Exiting")
        exit()
    
    cam.location.x = tx
    cam.location.y = ty
    cam.location.z = tz

    cam.rotation_mode = 'XYZ'
    cam.rotation_euler[0] = rx
    cam.rotation_euler[1] = ry
    cam.rotation_euler[2] = rz
    cam.data.angle = fov  
    # cam.data.lens = 28
    # cam.data.sensor_width = 36
    # cam.data.clip_start = 0.01


def log_iteration(params,error,iteration):
    out_path = os.path.join(RENDER_OUTPUT_PATH, "it_{:02d}_error_{:.2f}.png".format(iteration, error))
    out_path_json = os.path.join(RENDER_OUTPUT_PATH, "it_{:02d}_error_{:.2f}.json".format(iteration, error))
    trn_params = params.tolist()
    with open(out_path_json, "w+") as f:
        json.dump({"params": trn_params, "iteration": iteration, "error": error}, f)
    bpy.data.images["Render Result"].save_render(filepath=out_path)


def set_scene(render_settings):
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'

    bpy.context.scene.eevee.taa_render_samples = render_settings["spp"]
    scaler = 1
    plane_size = 4 * scaler # Tamaño del escenario en Km   
    
    if "Cube" in bpy.data.meshes: # Borra el cubo inicial
        mesh = bpy.data.meshes["Cube"]
        print("removing mesh", mesh)
        bpy.data.meshes.remove(mesh)

    tex = bpy.data.textures.new("mdt", 'IMAGE')
#    img = bpy.data.images.load(render_settings["mdt_path"])
    img = bpy.data.images.load(MDT_PATH)
    img.colorspace_settings.name = 'Non-Color'
    tex.image = img
    tex.extension = 'EXTEND'

    bpy.ops.mesh.primitive_plane_add(size=plane_size, location=(0, 0, 0))

    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.subdivide(number_cuts=79)
    bpy.ops.mesh.subdivide(number_cuts=1)

    bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.ops.object.modifier_add(type='DISPLACE')
    bpy.context.object.modifiers['Displace'].texture = tex
    bpy.context.object.modifiers['Displace'].texture_coords = 'UV'
    #bpy.context.object.modifiers['Displace'].strength = 0
    bpy.context.object.modifiers['Displace'].strength = 32.768
    bpy.context.object.modifiers['Displace'].mid_level = 0

    bpy.ops.object.shade_smooth()

    obj = bpy.context.object

    ## Asignar imagen aérea como image texture del material del plano
 
    orto_img = bpy.data.images.load( ORTO_PATH)
    mat = bpy.data.materials['Material']
    mat.use_nodes = True
    obj.data.materials.append(mat) 
    nodes = mat.node_tree.nodes
    node_texture = nodes.new(type='ShaderNodeTexImage')
    node_texture.image = orto_img
    links = mat.node_tree.links
    link = links.new(node_texture.outputs[0], nodes.get("Principled BSDF").inputs[0])
#    nodes.get("Principled BSDF").inputs[3].default_value = 1 # IOR
    nodes.get("Principled BSDF").inputs["IOR"].default_value = 1 # IOR
    nodes.get("Principled BSDF").inputs["Roughness"].default_value = 10000 # IOR
    nodes.get("Principled BSDF").inputs["Specular"].default_value = 0 # IOR

    camera = bpy.data.objects.get('Camera')
    camera.location[0] = 0.01
    camera.location[1] = 0
#    camera.location[2] = 1.0
#    camera.rotation_euler[0] = math.radians(90) 
#    camera.rotation_euler[1] = 0 * math.pi / 180
#    camera.rotation_euler[2] = 0 * math.pi / 180
    camera.data.lens = render_settings["focal_length"]
    camera.data.sensor_width = render_settings["sensor_width"]
    camera.data.clip_start = 0.01
    bpy.context.scene.render.resolution_x = RES_X
    bpy.context.scene.render.resolution_y = RES_Y

def render_to_image(path, resolution = (RES_X, RES_Y)):
    # redirect output to log file, to hide all the printing blender does for each rendering 
    
    scene = bpy.context.scene
    #scene.camera = bpy.data.objects[CAMERA_NAME]
    scene.render.filepath = path
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    bpy.ops.render.render(write_still=True)
    
    # disable output redirection

    return

def empty_penalty(image, threshold=0.001, weight=100000000.0):
    """
    image: numpy array in [0,1], shape (H, W, 3)
    threshold: minimum acceptable variance
    weight: how strong the penalty should be
    """
    var = np.var(image)
    
    #print("Image variance {0} threshold {1}".format(var, threshold))
    if var < threshold:
        return weight * (threshold - var)

    return 0.0


import math
def loss_function(it,target):

    mkpts_0, mkpts_1, conf = RDD.match(it, target,resize=256)
    if len(mkpts_0) > 0:
        l2 = l2_distance(mkpts_0,mkpts_1) / (len(mkpts_0))    
        ep = empty_penalty(it,weight=INFINITY)
        loss = ep + l2
        print(loss)
        return loss
    else:
        return INFINITY
def loss_function_wrapper(params):
    """
    CMA-ES objective: given camera params, render and compute loss vs target.
    """
    set_camera_from_params(params)
    img_p = os.path.join(RENDER_OUTPUT_PATH,"it.png")
    _ = render_to_image(img_p)
    rendered = cv2.imread(img_p)
    loss = loss_function(rendered,TARGET_IMG)
    print("*************ATTENTION PLEASE*******************")
    print(loss)
    return loss
    # Simple MSE loss


def get_boundaries():
    # define lower and upper bounds for the parameters
    lower_bounds = [0.0, -math.radians(180), -math.radians(180), -math.radians(180), math.radians(2)]   
    upper_bounds = [3.0, math.radians(180), math.radians(180), math.radians(180),math.radians(60)]
    return [lower_bounds, upper_bounds]

def l2_distance(a, b):
    return np.linalg.norm(a - b)

def draw_matches(ref_points, dst_points, img0, img1):
    
    # Prepare keypoints and matches for drawMatches function
    keypoints0 = [cv2.KeyPoint(p[0], p[1], 1000) for p in ref_points]
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 1000) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(ref_points))]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img0, keypoints0, img1, keypoints1, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches


def draw_points(points, img):
    for p in points:
        cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
        
    return img


def loss(img1,img2):
    start = time()
    mkpts_0, mkpts_1, conf = RDD.match(im0, im1, resize=256)
    print(f"Found {len(mkpts_0)} matches in {time()-start:.2f} seconds")
     
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if it's an image by trying to read it
        img = cv2.imread(file_path)

        if img is not None:   # Only append valid images
            images.append(img)
        
    return (images,os.listdir(folder_path))


def gen_images02(dictionary,dim1_name,dim2_name,dim3_name,dim1s,dim2s,dim3s,init_params,parent_folder):
    #render_settings = json.load(open(os.path.join(GLOBAL_PATH, "render_settings.json"), "r"))
    #set_scene(render_settings=render_settings)
    for val1 in dim1s:
        for val2 in dim2s:
            for val3 in dim3s:
                params = init_params.copy()
                params[dictionary[dim1_name]["idx"]] += val1
                params[dictionary[dim2_name]["idx"]] += val2
                params[dictionary[dim3_name]["idx"]] += val3
                set_camera_from_params(params)
                img_p = os.path.join(parent_folder,"it_{0}_{1}_{2}_{3}_{4}_{5}.png".format(dim1_name,val1,dim2_name,val2,dim3_name,val3))
                if DEBUG:
                    print(params)
                _ = render_to_image(img_p)


def gen_images(dictionary,dim1_name,dim2_name,dim1s,dim2s,init_params,parent_folder):
    #render_settings = json.load(open(os.path.join(GLOBAL_PATH, "render_settings.json"), "r"))
    #set_scene(render_settings=render_settings)
    for val1 in dim1s:
        for val2 in dim2s:
            params = init_params.copy()
            params[dictionary[dim1_name]["idx"]] += val1
            params[dictionary[dim2_name]["idx"]] += val2
            set_camera_from_params(params)
            img_p = os.path.join(parent_folder,"it_{0}_{1}_{2}_{3}.png".format(dim1_name,val1,dim2_name,val2))
            if DEBUG:
                print(params)
            
            rendered = render_to_image(img_p)

def process_filename(s,dim1_name,dim2_name):
    #match = re.search(r"h_([0-9.]+)_z_([0-9.]+)", s)
    match = re.search("{0}_([0-9.]+)_{1}_([0-9.]+)".format(dim1_name,dim2_name), s)

    if match:
        dim1 = float(match.group(1))
        dim2 = float(match.group(2)[:-1])
        return dim1, dim2
    else:
        return None,None
def process_filename02(s,dim1_name,dim2_name,dim3_name):
    #match = re.search(r"h_([0-9.]+)_z_([0-9.]+)", s)
    match = re.search("{0}_([0-9.]+)_{1}_([0-9.]+)_{2}_([0-9.]+)".format(dim1_name,dim2_name,dim3_name), s)

    if match:
        dim1 = float(match.group(1))
        dim2 = float(match.group(2))
        dim3 = float(match.group(3)[:-1])
        return dim1, dim2,dim3
    else:
        return None,None

def gen_database02(target_img_p,view_expl_folder_p,dim1_name,dim2_name,dim3_name):
 
    if DEBUG:
        plt.figure(figsize=(12,12))
 
    #df = pd.DataFrame(columns=["name", "loss", "lenght", dim1_name, dim2_name])
    #get the rendered images
    images,names = load_images_from_folder(view_expl_folder_p)
    orig = cv2.imread(target_img_p)
    iteration = 0
    rows = []
    for img in images:
#        mkpts_0, mkpts_1, conf = RDD.match_dense(img, orig, resize=256, anchor='mnn')
        mkpts_0, mkpts_1, conf = RDD.match(img, orig, resize=256)
        if len(mkpts_0) > 0:
            l2 = l2_distance(mkpts_0,mkpts_1) / len(mkpts_0)
        else:
            l2 = np.nan
        dim1,dim2,dim3 = process_filename02(names[iteration],dim1_name,dim2_name,dim3_name)
        #print("For image {0} we have a loss function of {1} with shape {2}".format(names[iteration],l2,mkpts_0.shape))
        rows.append({"name": names[iteration],"loss":l2,"lenght":len(mkpts_0), dim1_name: dim1, dim2_name: dim2,dim3_name: dim3 })
#        df.loc[len(df)] = [names[iteration],l2,len(mkpts_0),dim1,dim2]
        if DEBUG:
            canvas = draw_matches(mkpts_0,mkpts_1,img,orig)
            plt.imshow(canvas[..., ::-1])
            plt.savefig("loss_exploration/match_{0}".format(names[iteration]))
            #plt.clear()
            #plt.show()
        iteration += 1
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_FOLDER,"df_{0}_vs_{1}_vs_{2}.csv".format(dim1_name,dim2_name,dim3_name)))


def gen_database(target_img_p,view_expl_folder_p,dim1_name,dim2_name):
 
    if DEBUG:
        plt.figure(figsize=(12,12))
 
    #df = pd.DataFrame(columns=["name", "loss", "lenght", dim1_name, dim2_name])
    #get the rendered images
    images,names = load_images_from_folder(view_expl_folder_p)
    orig = cv2.imread(target_img_p)
    iteration = 0
    rows = []
    for img in images:
#        mkpts_0, mkpts_1, conf = RDD.match_dense(img, orig, resize=256, anchor='mnn')
        mkpts_0, mkpts_1, conf = RDD.match(img, orig, resize=256)
        if len(mkpts_0) > 0:
            l2 = l2_distance(mkpts_0,mkpts_1) / len(mkpts_0)
        else:
            l2 = np.nan
        dim1,dim2 = process_filename(names[iteration],dim1_name,dim2_name)
        #print("For image {0} we have a loss function of {1} with shape {2}".format(names[iteration],l2,mkpts_0.shape))
        rows.append({"name": names[iteration],"loss":l2,"lenght":len(mkpts_0), dim1_name: dim1, dim2_name: dim2 })
#        df.loc[len(df)] = [names[iteration],l2,len(mkpts_0),dim1,dim2]
        if DEBUG:
            canvas = draw_matches(mkpts_0,mkpts_1,img,orig)
            plt.imshow(canvas[..., ::-1])
            plt.savefig("loss_exploration/match_{0}".format(names[iteration]))
            #plt.clear()
            #plt.show()
        iteration += 1
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_FOLDER,"df_{0}_vs_{1}.csv".format(dim1_name,dim2_name)))

def get_candidates02(dim1_name,dim2_name,dim3_name):
    df = pd.read_csv(os.path.join(OUTPUT_FOLDER,"df_{0}_vs_{1}_vs_{2}.csv".format(dim1_name,dim2_name,dim3_name)))
    df['avg_keypoints'] = df.groupby(dim1_name)['lenght'].transform('mean')
    df = df.sort_values('avg_keypoints')
    avg_kpts = df['avg_keypoints'].unique()
    #set nans to zero, then sort again the array to ensure that they are last
    avg_kpts[np.isnan(avg_kpts)] = 0
    avg_kpts = np.sort(avg_kpts)
    # select the rows with the highest avg of keypoint found for dim1
    min_avg = avg_kpts[-2:-1]
    #now all the rows that have the two highest values
    df_filtered = df[df["avg_keypoints"] > float(min_avg[0])]
    min_index = df_filtered["loss"].idxmin()
    min_row = df_filtered.loc[min_index]
    candidate_dim1 = min_row[dim1_name]
    candidate_dim2 = min_row[dim2_name]
    candidate_dim3 = min_row[dim3_name]
    return candidate_dim1,candidate_dim2,candidate_dim3



def get_candidates(dim1_name,dim2_name):
    df = pd.read_csv(os.path.join(OUTPUT_FOLDER,"df_{0}_vs_{1}.csv".format(dim1_name,dim2_name)))
    df['avg_keypoints'] = df.groupby(dim1_name)['lenght'].transform('mean')
    df = df.sort_values('avg_keypoints')
    avg_kpts = df['avg_keypoints'].unique()
    #set nans to zero, then sort again the array to ensure that they are last
    avg_kpts[np.isnan(avg_kpts)] = 0
    avg_kpts = np.sort(avg_kpts)
    # select the rows with the highest avg of keypoint found for dim1
    min_avg = avg_kpts[-2:-1]
    #now all the rows that have the two highest values
    df_filtered = df[df["avg_keypoints"] > float(min_avg[0])]
    if DEBUG:
        print("Selected filter value " + float(min_avg[0]))
        print(len(df_filtered))
        print(avg_kpts)
    #now select the best candidate for starting the optimization
    df_filtered = df_filtered.sort_values('loss')
    df_filtered = df_filtered.reset_index()
    candidate_dim1 = df_filtered.loc[0][dim1_name]
    candidate_dim2 = df_filtered.loc[0][dim2_name]
    return candidate_dim1,candidate_dim2



def get_starting_point02(dictionary,dim1_name,dim2_name,dim3_name,init_params):


    
    dim1s = dictionary[dim1_name]["space"]
    dim2s = dictionary[dim2_name]["space"]
    dim3s = dictionary[dim3_name]["space"]

    # Gen images
    
    view_expl_folder_p = os.path.join(OUTPUT_FOLDER,"view_exploration_{0}_vs_{1}_vs_{2}".format(dim1_name,dim2_name,dim3_name))
    mkfolder(view_expl_folder_p)

    gen_images02(dictionary,dim1_name,dim2_name,dim3_name,dim1s,dim2s,dim3s,init_params,view_expl_folder_p)
    
    # Estimate distances
    
    gen_database02(TARGET_IMAGE_PATH,view_expl_folder_p,dim1_name,dim2_name,dim3_name)
    
    # Find promising candidates
    c_dim1,c_dim2,c_dim3 = get_candidates02(dim1_name,dim2_name,dim3_name)
    
    return c_dim1, c_dim2, c_dim3


def get_starting_point(dictionary,dim1_name,dim2_name,init_params):


    
    dim1s = dictionary[dim1_name]["space"]
    dim2s = dictionary[dim2_name]["space"]

    # Gen images
    
    view_expl_folder_p = os.path.join(OUTPUT_FOLDER,"view_exploration_{0}_vs_{1}".format(dim1_name,dim2_name))
    mkfolder(view_expl_folder_p)

    gen_images(dictionary,dim1_name,dim2_name,dim1s,dim2s,init_params,view_expl_folder_p)
    
    # Estimate distances
    
    gen_database(TARGET_IMAGE_PATH,view_expl_folder_p,dim1_name,dim2_name)
    
    # Find promising candidates

    c_dim1,c_dim2 = get_candidates(dim1_name,dim2_name)
    
    return c_dim1, c_dim2

def get_min_height():
    #set origing really high (9.0 km)
    origin = Vector((0.0, 0, 9.0))
    # shoot ray downward
    direction = Vector((0, 0, -1))

    # Get evaluated depsgraph
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    # Perform ray cast
    result, location, normal, index, obj, matrix = bpy.context.scene.ray_cast(
        depsgraph,
        origin,
        direction
    )

    if result:
        print("Hit object:", obj.name)
        print("Location:", location)
        print("Normal:", normal)
        return location[2]
    else:
        print("No hit")
        return 0.1

def main():
    start = time()

    # Initial guess for camera parametersff
    # [tz, rx, ry, rz, fov]

    #Feature points extractor
    RDD_model = build(weights='./weights/RDD-v2.pth')
    RDD_model.eval()
    RDD = RDD_helper(RDD_model)
    
    #gen folder system
    mkfolder(os.path.join(LOCATIONS_PATH,LOCATION,"cma_optim"))
    mkfolder(RENDER_OUTPUT_PATH)
    
    x0 = [0.0, math.radians(0.0), math.radians(0.0), math.radians(0.0), math.radians(80.0)]

    # Define research space
    dictionary = {}
    render_settings = json.load(open(os.path.join("./render_settings.json"), "r"))
    set_scene(render_settings=render_settings)
    min_h = get_min_height() + 0.1 #find the lowest point and then add 100 meters to avoid terrain collision
    dictionary["height"] = {"space" : np.linspace(min_h,2,10),                  "idx" : 0 }
    dictionary["x_rot"] =  {"space" : np.linspace(math.radians(0.0),math.radians(90),4), "idx" : 1 } #We only search in one hemisphere since for negative values we would have chosen opposite values of z 
    dictionary["y_rot"] =  {"space" : np.linspace(0.0,math.radians(360),36), "idx" : 2 }
    dictionary["z_rot"] =  {"space" : np.linspace(0.0,math.radians(360),36), "idx" : 3 }
    dictionary["fov"] =    {"space" : np.linspace(math.radians(10),math.radians( 70), 6), "idx" : 4 }
    # Grid search in 3 dimensions (height,z_rot,x_rot)
    h, z_rot, x_rot = get_starting_point02(dictionary,dim1_name="height", dim2_name="z_rot",dim3_name="x_rot", init_params=x0)
    
    #set to zero fov and rotation angle as we are going to explore these dimensions
    x1 = [h, x_rot, math.radians(0.0), z_rot, math.radians(0.0)]
    #refine the search over the x-dimension, by increasing the number of samples used 
    dictionary["x_rot"] =  {"space" : np.linspace(math.radians(0.0),math.radians(180),18), "idx" : 1 } #We only search in one hemisphere since for negative values we would have chosen opposite values of z 
    #Grid search in 2 dimensions (fov,x_rot)
    fov, extra_rot = get_starting_point(dictionary,dim1_name="fov", dim2_name="x_rot", init_params=x1)
    # Since usually the photos have little tilt (rotation along the y) we let the final optimization explore this dimension as well, by adding a small rotation to the x_rot value obtained from the grid search, to give some space for the optimization to explore this dimension as well.
    x2 = [h,x_rot+extra_rot,math.radians(10),z_rot,fov]

    #reset the scene
    render_settings = json.load(open(os.path.join("./render_settings.json"), "r"))
    set_scene(render_settings=render_settings)
    # Run CMA-ES
#    params2 = [1.5, math.radians(90), math.radians(0.0), math.radians(15.0), math.radians(32)]
    set_camera_from_params(x2)
    #init_path = os.path.join(RENDER_OUTPUT_PATH,"init_scene.png")
    init_path = os.path.join(LOCATIONS_PATH,LOCATION,LOCATION + "_init.png")

    _ = render_to_image(init_path)
    
    loss = loss_function(cv2.imread(init_path),TARGET_IMG)
    print("INIT LOSS {0}".format(loss))
    print(x2)
    #now that we are reasonably close to the solution, displace the camera a bit and slightly rotate along the y rot axis so the optimization has space to optimize these values as well.
    # we are effectively increasing the search space by two dimensions, but this should account for small displacements in the x-y plane
    x3 = [0.0,0.0,h,x_rot+extra_rot,math.radians(0),z_rot,fov]
    print(x3)
    optim_loop(x3)
    final_path = os.path.join(LOCATIONS_PATH,LOCATION,LOCATION + "_final.png")

    _ = render_to_image(final_path)

    print(f"Parameters found in {time()-start:.4f} seconds")

# I've used 20% for the first optimization, then changed it to 10 for the second search, to limit params exploration
def get_bnds(val,percentage=20):
    l_b = val - val/100*percentage
    u_b = val + val/100*percentage
    return (l_b,u_b)

def log_callback(intermediate_result):
    # xk: current best solution
    # convergence: float indicating population convergence (0=perfect)
    print(f"Current best x: {intermediate_result.x}, with loss: {intermediate_result.fun}")
    out_path = os.path.join(RENDER_OUTPUT_PATH, "it_{:02d}_error_{:.2f}.png".format(0,50))
    out_path_json = os.path.join(RENDER_OUTPUT_PATH,"it_{:02d}_error_{:.2f}.json".format(0,50))
    
    xk = intermediate_result.x
    trn_params = xk.tolist()
    trn_params[1] = math.degrees(trn_params[1])
    trn_params[2] = math.degrees(trn_params[2])
    trn_params[3] = math.degrees(trn_params[3])
    trn_params[4] = math.degrees(trn_params[4])
    with open(out_path_json, "w+") as f:
        json.dump({"params": trn_params}, f)
    bpy.data.images["Render Result"].save_render(filepath=out_path)

def get_initial_pop(x0,bounds,n_individuals=15,dim=5):
    pop = np.random.rand(n_individuals, dim)
    for i in range(dim):
        pop[:, i] = bounds[i][0] + pop[:, i]*(bounds[i][1]-bounds[i][0])
    # insert our best candidate
    pop[n_individuals-1] = x0
    return pop

def optim_loop(x0):
    import numpy as np
    from scipy.optimize import differential_evolution

    # Objective function: simple 2D function
    
    # Bounds for x0 and x1

    # Run Differential Evolution
    if len(x0) == 5:
        z =     float(x0[0])
        x_rot = float(x0[1])
        y_rot = float(x0[2])
        z_rot = float(x0[3])
        fov =   float(x0[4])
        bounds = [get_bnds(z),get_bnds(x_rot),get_bnds(y_rot,percentage=100),get_bnds(z_rot),get_bnds(fov)]
    else:
        x =     float(x0[0])
        y =     float(x0[1])
        z =     float(x0[2])
        x_rot = float(x0[3])
        y_rot = float(x0[4])
        z_rot = float(x0[5])
        fov =   float(x0[6])
        bounds = [(-0.2,0.2),(-0.2,0.2),get_bnds(z),get_bnds(x_rot),(math.degrees(-20),math.degrees(20)),get_bnds(z_rot),get_bnds(fov)]
    
    
    
    initial_pop = get_initial_pop(x0,bounds=bounds,n_individuals=5,dim=len(x0))

    result = differential_evolution(loss_function_wrapper, bounds, maxiter=230,
    popsize=10,
    mutation=[0.65,1.1],
    recombination=0.8,
    seed=42,
    callback=log_callback,
    init = initial_pop,
    strategy='best1bin',
    tol=1e-6)

    print("Optimal solution:", result.x)
    print("Function value:", result.fun)


if __name__ == "__main__":
    main()
    print("Script correctly terminated.")
