from concurrent.futures import process
import cv2
import trimesh
from trimesh.visual import texture, TextureVisuals
from trimesh import Trimesh
from PIL import Image
import numpy as np

img = Image.open("texture.jpg")

# tex = TextureVisuals(image = img)
name_model ="result_Lada2_256.obj"

with open(name_model, 'r') as f:
    mesh_uv = trimesh.exchange.obj.load_obj(f)




# uv = mesh_uv['visual'].uv
# print(uv)
# exit()
mesh = trimesh.load(name_model, process = False)

material = trimesh.visual.texture.SimpleMaterial(image=img)

# color_visuals = trimesh.visual.TextureVisuals(uv=uv, image=img, material=material)
mesh=trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, validate=True, process=False)
mesh.show()


# voxelized_mesh = mesh.voxelized(0.01).hollow().as_boxes()

# s = trimesh.Scene()
# s.add_geometry(voxelized_mesh)
# s.show()


# cloud_of_points = trimesh.points.PointCloud(mesh.vertices, colors= (0,0,0,255))
# s = trimesh.Scene()
# s.add_geometry(cloud_of_points)
# s.show()