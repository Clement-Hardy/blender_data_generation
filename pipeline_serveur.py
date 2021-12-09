#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 18:04:25 2021

@author: hardy216
"""
import bpy
import numpy as np
import os
import time
import shutil
import math
import bmesh



def fibonacci_sphere(samples, rayon=1, x_center=0, y_center=0, z_center=0):

    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((rayon*x+x_center, rayon*y+y_center, rayon*z+z_center))

    return np.array(points)



def read_material(folder, provider, texture_index):
    global global_texture_index
    
    if provider=="ambientcg":
        path = os.path.join(path_texture_ambientcg, folder)
        filename = folder.split("-JPG")[0]
        file_img_color = os.path.join(path, "{}_Color.jpg".format(filename))
        file_img_rough = os.path.join(path, "{}_Roughness.jpg".format(filename))
        file_img_metal = os.path.join(path, "{}_Metalness.jpg".format(filename))
        file_img_displacement = os.path.join(path, "{}_Displacement.jpg".format(filename))
        
        
    name_mat = "mat_{}".format(texture_index)
    mat = bpy.data.materials.new(name=name_mat)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    
    
    texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    texImage.image = bpy.data.images.load(file_img_color)
    mat.node_tree.links.new(bsdf.inputs['Base Color'],
                            texImage.outputs['Color'])
    
    texImage_rough = mat.node_tree.nodes.new('ShaderNodeTexImage')
    texImage_rough.image = bpy.data.images.load(file_img_rough)
    texImage_rough.image.colorspace_settings.name = 'Non-Color'
    mat.node_tree.links.new(bsdf.inputs['Roughness'],
                            texImage_rough.outputs['Color'])
                            
                            
    if os.path.exists(file_img_metal):
        texImage_metal = mat.node_tree.nodes.new('ShaderNodeTexImage')
        texImage_metal.image = bpy.data.images.load(file_img_metal)
        texImage_metal.image.colorspace_settings.name = 'Non-Color'
        mat.node_tree.links.new(bsdf.inputs['Metallic'],
                                texImage_rough.outputs['Color'])
                                
    img_displacement = bpy.data.images.load(file_img_displacement)
    return mat, img_displacement




def apply_displacement(obj, img_displacement):
    bpy.context.view_layer.objects.active = obj
    """
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=1)
    bpy.ops.object.mode_set(mode="OBJECT")

    mod = obj.modifiers.new("", 'SUBSURF')
    mod.levels = 1

    mod.subdivision_type = "SIMPLE"
    """
    area = sum([bpy.context.active_object.data.polygons[i].area for i in range(len(bpy.context.active_object.data.polygons))])
    
    #print(len(bpy.context.active_object.data.polygons))
    tex = bpy.data.textures.new(name="Tex_Altura", type="IMAGE")
    
    tex.image = img_displacement
    tex.extension = 'EXTEND'

    mod = obj.modifiers.new("", 'DISPLACE')
    mod.strength = area/1000
    mod.mid_level = 0
    mod.texture_coords="UV"
    mod.texture = tex
    #bpy.ops.object.convert(target='MESH', keep_original=False)
    
    

def apply_random_texture():
    for material in bpy.data.materials:
        material.user_clear()
        bpy.data.materials.remove(material)
        
    texture_index = 0
    for obj in bpy.data.objects:
        if obj.name!="Vide" and obj.name!="Camera" and obj.name!="Spot" and len(obj.children)==0:
            folder_feature = np.random.choice(possible_features_ambientcg)
            
            mat, img_displacement = read_material(provider="ambientcg",
                                                      folder=folder_feature,
                                                      texture_index=texture_index)
            
                
            texture_index+=1
            
            
            bpy.context.view_layer.objects.active = obj
            obj = bpy.context.view_layer.objects.active
            bpy.ops.object.editmode_toggle()
            bpy.ops.mesh.select_all(action='SELECT') # for all faces
            #bpy.ops.uv.smart_project(angle_limit=66, island_margin = 0.02)
            bpy.ops.object.editmode_toggle()
            obj.select_set(False)
            if obj.data.materials:
                obj.data.materials[0] = mat
            else:
                obj.data.materials.append(mat)
                
            #apply_displacement(obj, img_displacement)
            
    
    
def location(coordinates):

    x = [vertex.x for vertex in coordinates]
    y = [vertex.y for vertex in coordinates]
    z = [vertex.z for vertex in coordinates]

    return {"x": np.mean(x), "y": np.mean(y), "z": np.mean(z)}

    
def translation_obj(obj, x_change, y_change, z_change):
     obj.location[0]+=x_camera
     obj.location[1]+=y_camera
     obj.location[2]+=z_camera
     
def change_pos_obj(obj, x, y, z):
    obj.location[0] = x
    obj.location[1] = y
    obj.location[2] = z   
    

def put_rotation_zero(obj):
    obj.rotation_mode = "XYZ"
    obj.rotation_euler[0] = 0
    obj.rotation_euler[1] = 0
    obj.rotation_euler[2] = 0
    
    if len(obj.children)!=0:
        for obj1 in obj.children:            
            put_rotation_zero(obj1)

def set_to_origin_obj(obj, rho):
    coordinates = []
    for object in bpy.data.objects:
        if hasattr(object, "data") and hasattr(object.data, "vertices"):
            object.rotation_mode = "XYZ"
            object.rotation_euler[0] = 0
            object.rotation_euler[1] = 0
            object.rotation_euler[2] = 0
            coordinates += [vertex.co for vertex in object.data.vertices]
    
    
    transform = location(coordinates)
    for object in bpy.data.objects:
        if hasattr(object, "data") and hasattr(object.data, "vertices"):
            object.rotation_mode = "XYZ"
            object.location[0]=-transform["x"]*object.scale[0]
            object.location[1]=-transform["y"]*object.scale[1]
            object.location[2]=-transform["z"]*object.scale[2]

    
    if len(obj.children)==0:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
    obj.location[0]=- rho
    obj.location[1]=0
    obj.location[2]=0
    
            
    
def copy_ob(ob, parent,  collection=bpy.context.collection):
    # copy ob
    copy = ob.copy()
    copy.parent = parent
    copy.matrix_parent_inverse = ob.matrix_parent_inverse.copy()
    # copy particle settings
    for ps in copy.particle_systems:
        ps.settings = ps.settings.copy()
    collection.objects.link(copy)
    return copy
    
def tree_copy(ob, parent, levels=5):
    def recurse(ob, parent, depth):
        if depth > levels: 
            return
        copy = copy_ob(ob, parent)
        
        for child in ob.children:
            recurse(child, copy, depth + 1)
    recurse(ob, ob.parent, 0)
        
            
def remove_object(object_name):
    objs = bpy.data.objects
    if object_name in objs.keys():
        for children in objs[object_name].children:
            remove_object(children.name)
        objs.remove(objs[object_name], do_unlink=True)


def get_object_name():
    for object in bpy.data.objects:
        if object.name!="Vide" and object.name!="Camera" and object.name!="Spot" and object.parent is None:
            return object.name
            
            
def take_picture_view(obj, folder_name_save, path_dataset, coord_lamps, x_rotate, y_rotate, z_rotate, rho):
    
                
    depth = True
    
    nodetree = bpy.context.scene.node_tree
    
    nodetree.links.new(nodetree.nodes["Normaliser"].outputs["Value"],
                               nodetree.nodes['Sortie fichier.001'].inputs["Image"])
    nodetree.links.new(nodetree.nodes['Calques de rendu'].outputs['Normal'],
                                   nodetree.nodes['Sortie fichier.002'].inputs["Image"])
    nodetree.links.new(nodetree.nodes['Calques de rendu'].outputs['Alpha'],
                                   nodetree.nodes['Sortie fichier.003'].inputs["Image"])
        
    diff_theta = np.pi/2
    start_theta=-np.pi+diff_theta
    end_theta=-np.pi-diff_theta
        
    diff_alpha = np.pi/2
    start_alpha=-np.pi/2-diff_alpha
    end_alpha=-np.pi/2+diff_alpha
                
    already_done = True
    for coord in coord_lamps:
        random_angle_add = np.random.rand(3)*0.01
                        
        x = coord[0] + random_angle_add[0]
        y = coord[1] + random_angle_add[1]
        z = coord[2] + random_angle_add[2]
                    
        if np.abs(x)<1e-10:
            x = 0.
        if np.abs(y)<1e-10:
            y = 0.
        if np.abs(z)<1e-10:
            z = 0.
        spot = bpy.data.objects["Spot"]
        spot.location[0] = x
        spot.location[1] = y
        spot.location[2] = z
                        
        file_input = os.path.join(path_dataset, "img", "Image0001.{}".format(format))
        folder = os.path.join(path_dataset, "img", folder_name_save, "view_{}_{}_{}".format(x_rotate, y_rotate, z_rotate))
        if not os.path.exists(folder):
            os.makedirs(folder)
            already_done = False
                            
        if not already_done:
            file_output = os.path.join(folder, "Image_{}_{}_{}_{}.{}".format(x, y, z, rho, format))
            if not os.path.exists(file_output):
                b=bpy.ops.render.render()
                if len(nodetree.nodes['Sortie fichier.001'].inputs["Image"].links)==1:
                    nodetree.links.remove(nodetree.nodes['Sortie fichier.001'].inputs["Image"].links[0])
                if len(nodetree.nodes['Sortie fichier.002'].inputs["Image"].links)==1:
                    nodetree.links.remove(nodetree.nodes['Sortie fichier.002'].inputs["Image"].links[0])
                if len(nodetree.nodes['Sortie fichier.003'].inputs["Image"].links)==1:
                    nodetree.links.remove(nodetree.nodes['Sortie fichier.003'].inputs["Image"].links[0])
                            
                #time.sleep(0.01)
                shutil.move(file_input, file_output) 
                
                if depth:
                    file_scan_input = os.path.join(path_dataset, "depth", "Image0001.{}".format(format))
                    folder = os.path.join(path_save, "depth", folder_name_save)
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                        
                    file_scan_output = os.path.join(folder, "view_{}_{}_{}.{}".format(x_rotate, y_rotate, z_rotate, format))
                    shutil.move(file_scan_input, file_scan_output)
        
                    file_normal_input = os.path.join(path_dataset, "normal", "Image0001.{}".format(format))
                    folder = os.path.join(path_save, "normal", folder_name_save)
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                        
                    file_normal_output = os.path.join(folder, "view_{}_{}_{}.{}".format(x_rotate, y_rotate, z_rotate, format))
                    shutil.move(file_normal_input, file_normal_output)
                                
                    depth = False
      


def load_object(file, folder_name_save, path_dataset, params):
    if "gltf" in file:
        bpy.ops.import_scene.gltf(filepath=file)
    elif ".obj" in file:
        bpy.ops.import_scene.obj(filepath=file)
    elif ".stl" in file:
        bpy.ops.import_mesh.stl(filepath=file)

    size_min = 0
    for obj in bpy.data.objects:
        
        if obj.name!="Vide" and obj.name!="Camera" and obj.name!="Spot" and hasattr(obj,"dimensions"):
            if obj.parent is None:
                bpy.context.view_layer.objects.active = obj
            if size_min<obj.dimensions[0]:
                size_min = obj.dimensions[0]
            if size_min<obj.dimensions[1]:
                size_min = obj.dimensions[1]
            if size_min<obj.dimensions[2]:
                size_min = obj.dimensions[2]               
                
    scale = 1./size_min

    obj = bpy.context.object

    if scale!=1:
        obj.scale[0]*=scale
        obj.scale[1]*=scale
        obj.scale[2]*=scale    
    
    put_rotation_zero(obj)
    set_to_origin_obj(obj=obj, rho=params["rho"])
    
    
    bpy.data.objects['Vide'].location[0] = -params["rho"]
    bpy.data.objects['Vide'].location[1] = 0
    bpy.data.objects['Vide'].location[2] = 0
    
    for material in bpy.data.materials:
        material.user_clear()
        bpy.data.materials.remove(material)
    
          
def process_file(file, folder_name_save, path_dataset):
    
    params = {}
    params["rho"] = 4

    distance_max_camera_lamp = 0.2
    coord_lamps = fibonacci_sphere(samples=5000, rayon=params["rho"],
                                   x_center=-params["rho"])
    coord_lamps = coord_lamps[coord_lamps[:,0]>-distance_max_camera_lamp]
    
    
    #bpy.context.collection.objects.link(obj_save)
    
    for x_rotate in np.linspace(0, 2*np.pi, 3, endpoint=False):
        for y_rotate in np.linspace(0, 2*np.pi, 3, endpoint=False):
            for z_rotate in np.linspace(0, 2*np.pi, 3, endpoint=False):
                folder = os.path.join(path_dataset, "img", folder_name_save, "view_{}_{}_{}".format(x_rotate, y_rotate, z_rotate))
                if not os.path.exists(folder):
                    load_object(file=file, folder_name_save=folder_name_save, path_dataset=path_dataset,
                                params=params)
                
                    obj = bpy.context.object
                
                    obj.rotation_euler[0] = x_rotate
                    obj.rotation_euler[1] = y_rotate
                    obj.rotation_euler[2] = z_rotate
                
                    camera = bpy.data.objects["Camera"]
                    camera.location[0] = 0
                    camera.location[1] = 0
                    camera.location[2] = 0
                
                
                    apply_random_texture()
                
                
                    take_picture_view(obj=obj, folder_name_save=folder_name_save,
                                      path_dataset=path_dataset, coord_lamps=coord_lamps,
                                      x_rotate=x_rotate, y_rotate=y_rotate, z_rotate=z_rotate,
                                      rho=params["rho"])
                
                    for object in bpy.data.objects:
                        if object.name!="Vide" and object.name!="Camera" and object.name!="Spot":
                            remove_object(object.name)
                        
    
   
    
    objs = bpy.data.objects
    for object in bpy.data.objects:
        if object.name!="Vide" and object.name!="Camera" and object.name!="Spot":
            objs.remove(objs[object.name], do_unlink=True)
    


path = "/data/chercheurs/hardy216/data/blender_data_generation"
path_save = os.path.join(path, "blender_result")
path_obj = os.path.join(path, "blender_obj")

path_texture = os.path.join(path_obj, "texture")
path_texture_texturescom = os.path.join(path_texture, "TexturesCom")
path_texture_ambientcg = os.path.join(path_texture, "ambientcg")

possible_features_ambientcg = os.listdir(path_texture_ambientcg)

format = "exr"

print("Begin process object")

path_object = os.path.join(path_obj, "object")
for folder_object in os.listdir(path_object):
    file = os.path.join(path_object, folder_object, "scene.gltf")
    process_file(file=file,
                 folder_name_save=folder_object,
                 path_dataset=path_save)
                 
    print(folder_object)

"""
print("Begin process blobby David")
path_blobby_david = os.path.join(path_obj, "blobby")
for filename in np.random.choice(os.listdir(path_blobby_david), 10):
    if ".mtl" not in filename:
        file = os.path.join(path_blobby_david, filename)
        process_file(file=file,
                     folder_name_save=filename.split('.obj')[0],
                     path_dataset=path_save)          
"""
"""        
print("Begin process shape NetCore")
path_shapeNetCore = os.path.join(path_obj, "shapeNetCore")
for folder_object in np.random.choice(os.listdir(path_shapeNetCore), 20):
    file = os.path.join(path_shapeNetCore, folder_object, "model.obj")
    process_file(file=file,
                 folder_name_save=folder_object,
                 path_dataset=path_save)

print("Begin process shape NetSem")
path_shapeNetSem = os.path.join(path_obj, "shapeNetSem")
for filename in np.random.choice(os.listdir(path), 20):
    if ".mtl" not in filename:
        file = os.path.join(path, filename)
        process_file(file=file,
                     folder_name_save=filename.split('.obj')[0],
                     path_dataset=path_save)

print("Begin process ABC")
path_ABC = os.path.join(path_obj, "ABC")
for filename in np.random.choice(os.listdir(path_ABC), 50):
    if ".mtl" not in filename:
        file = os.path.join(path_ABC, filename)
        process_file(file=file,
                     folder_name_save=filename.split('.obj')[0],
                     path_dataset=path_save)
                     
print("Begin process Thingi10K")
path_Thingi10k = os.path.join(path_obj, "Thingi10K")
for filename in np.random.choice(os.listdir(path_Thingi10k), 20):
    file = os.path.join(path, filename)
    process_file(file=file,
                folder_name_save=filename.split('.stl')[0],
                path_dataset=path_save)
              
print("Begin process blob_Clement")
path_blob_Clement = os.path.join(path_obj, "blob_Clement")
for filename in np.random.choice(os.listdir(path_blob_Clement), 20):
    file = os.path.join(path_blob_Clement, filename)
    process_file(file=file,
                folder_name_save=filename.split('.obj')[0],
                path_dataset=path_save)                     
"""            
print("finish")
