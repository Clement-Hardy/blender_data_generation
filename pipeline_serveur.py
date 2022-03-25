import bpy
import numpy as np
import os, signal
import time
import shutil
import math
import cv2
import pandas as pd
import GPUtil
import uuid
import subprocess
import psutil
import argparse
from apscheduler.schedulers.background import BackgroundScheduler
import sys


def giga_in_bytes(giga):
    return giga*1000000000


def Randomwalk1D(n, max_value=100):
   n=n-1
   y = 0
   yposition = [0] 
   for i in range (1,n+1):
       step = np.random.uniform(0,1)
       if step>0.5 and y<0.8*max_value:
           y+=8
       elif step<0.5 and y>-0.8*max_value:
           y-=8
       elif step>((y+max_value)/(max_value*2)):  
           
           y += 8
       else: 
           y += -8
 
       yposition.append(y)
   return  pd.DataFrame(yposition)


def gen_mask(index, shape_img1=(1000,2000), shape_img2=1875):
    len_img = shape_img1[1]
    shape_img1 = int(shape_img1[0])
    shape_img2 = int(shape_img2)
    
    index = index.astype(np.uint32)
    mask = np.zeros((shape_img2, len_img))
    for i in range(len(index)):
        mask[index[i]:,i] = 1
    return mask


def cat(img1, img2, overlap=1/4):
    
    shape_img = (int(img2.shape[0]), int(img2.shape[1]))
    shape_img_first = int(img1.shape[0])
    img_last = img1[:(shape_img_first-shape_img[0]),:,:]
    img1 = img1[(shape_img_first-shape_img[0]):,:,:]
    
    shape_mask = shape_img[0]+int(shape_img[0]*(1-overlap))
    
    index = find_random_border(shape_img=shape_img,
                               overlap=overlap)
    mask = gen_mask(index=index,
                    shape_img1=shape_img,
                    shape_img2=shape_mask)
    
    result = np.zeros((mask.shape[0], mask.shape[1], img1.shape[-1]))
    
    x_min = 0
    x_max = shape_img[0]
    y_min = 0
    y_max = shape_img[1]
    mask1 = mask[x_min:x_max, y_min:y_max]
    result[x_min:x_max, y_min:y_max,:][mask1==0] = img1[mask1==0]
    
    x_min = int(shape_img[0]*(1-overlap))
    x_max = x_min+shape_img[0]
    
    mask1 = mask[x_min:x_max, y_min:y_max]
    result[x_min:x_max, y_min:y_max,:][mask1==1] = img2[mask1==1]
    result = np.concatenate((img_last, result))
    return result
    

        
def find_random_border(shape_img=(1000,2000), nb_try=1, overlap=1/4):
    max_over =int(shape_img[0]*overlap/2.1)

    lag = 5
    #df = pd.DataFrame(np.random.randn(shape_img[1]+lag) * sqrt(vol)).cumsum()
    df = Randomwalk1D((shape_img[1]+lag), max_value=max_over)
    data = (np.array(df.rolling(lag).mean()).flatten()[lag:]).astype(np.int16)
    if nb_try>50:
        data = np.zeros(shape_img[1])
    elif np.min(data)<=-max_over or np.max(data)>=max_over:
        nb_try+=1
        return find_random_border(shape_img=shape_img, nb_try=nb_try)
    line_border = int(shape_img[0]*(1-overlap/2))
    data+=line_border
    return data


def concatenate_imgs(imgs_color, imgs_rough, imgs_metal, imgs_normal, imgs_displacement):
    for i in range(1, len(imgs_color)):
        imgs_color[i] = cv2.resize(imgs_color[i],
                                 (imgs_color[0].shape[1], imgs_color[0].shape[0]))
        imgs_rough[i] = cv2.resize(imgs_rough[i],
                                  (imgs_color[0].shape[1], imgs_color[0].shape[0]))
        imgs_metal[i] = cv2.resize(imgs_metal[i],
                                  (imgs_color[0].shape[1], imgs_color[0].shape[0]))
        imgs_displacement[i] = cv2.resize(imgs_displacement[i],
                                         (imgs_color[0].shape[1], imgs_color[0].shape[0]))
        imgs_normal[i] = cv2.resize(imgs_normal[i],
                                     (imgs_color[0].shape[1], imgs_color[0].shape[0]))
    
    
    if np.sqrt(len(imgs_color))%1!=0:
        nb_img_axes = len(imgs_color)
    else:
        nb_img_axes= int(np.sqrt(len(imgs_color)))
    
    
    temp_color = cat(imgs_color[0], imgs_color[1])
    temp_rough = cat(imgs_rough[0], imgs_rough[1])
    temp_displacement = cat(imgs_displacement[0], imgs_displacement[1])
    temp_metal = cat(imgs_metal[0], imgs_metal[1])
    temp_normal = cat(imgs_normal[0], imgs_normal[1])
    idx = 2
    for i in range(2, nb_img_axes):
        temp_color=cat(temp_color, imgs_color[i])
        temp_rough=cat(temp_rough, imgs_rough[i])
        temp_displacement=cat(temp_displacement, imgs_displacement[i])
        temp_metal=cat(temp_metal, imgs_metal[i])
        temp_normal=cat(temp_normal, imgs_normal[i])
        idx+=1
    
    
    if np.sqrt(len(imgs_color))%1==0:
        temp_color = np.moveaxis(temp_color, 0, 1)
        temp_rough = np.moveaxis(temp_rough, 0, 1)
        temp_displacement = np.moveaxis(temp_displacement, 0, 1)
        temp_metal = np.moveaxis(temp_metal, 0, 1)
        temp_normal = np.moveaxis(temp_normal, 0, 1)
        
        for j in range(1, nb_img_axes):
            temp1_color = cat(imgs_color[idx], imgs_color[(idx+1)])
            temp1_rough = cat(imgs_rough[idx], imgs_rough[(idx+1)])
            temp1_displacement = cat(imgs_displacement[idx], imgs_displacement[(idx+1)])
            temp1_metal = cat(imgs_metal[idx], imgs_metal[(idx+1)])
            temp1_normal = cat(imgs_normal[idx], imgs_normal[(idx+1)])
            idx+=2
        
            for i in range(2, nb_img_axes):
                temp1_color=cat(temp1_color, imgs_color[idx])
                temp1_rough=cat(temp1_rough, imgs_rough[idx])
                temp1_displacement=cat(temp1_displacement, imgs_displacement[idx])
                temp1_metal=cat(temp1_metal, imgs_metal[idx])
                temp1_normal=cat(temp1_normal, imgs_normal[idx])
                idx+=1
            temp1_color = np.moveaxis(temp1_color, 0, 1)
            temp1_rough = np.moveaxis(temp1_rough, 0, 1)
            temp1_displacement = np.moveaxis(temp1_displacement, 0, 1)
            temp1_metal = np.moveaxis(temp1_metal, 0, 1)
            temp1_normal = np.moveaxis(temp1_normal, 0, 1)
                
            temp_color=cat(temp_color, temp1_color)
            temp_rough=cat(temp_rough, temp1_rough)
            temp_displacement=cat(temp_displacement, temp1_displacement)
            temp_metal=cat(temp_metal, temp1_metal)
            temp_normal=cat(temp_normal, temp1_normal)    
        
    return temp_color, temp_rough, temp_displacement, temp_metal, temp_normal
   
        
        
def read_material(provider, texture_index, nb_texture):
    global global_texture_index
    
    imgs_color = [] 
    imgs_rough = []
    imgs_metal = []
    imgs_displacement = []
    imgs_normal = []
    for i in range(nb_texture):
        if provider=="ambientcg":
            folder = np.random.choice(possible_features_ambientcg)
            path = os.path.join(path_texture_ambientcg, folder)
            filename = folder.split("-JPG")[0]
            file_img_color = os.path.join(path, "{}_Color.jpg".format(filename))
            file_img_rough = os.path.join(path, "{}_Roughness.jpg".format(filename))
            file_img_metal = os.path.join(path, "{}_Metalness.jpg".format(filename))
            file_img_normal = os.path.join(path, "{}_NormalGL.jpg".format(filename))
            file_img_displacement = os.path.join(path, "{}_Displacement.jpg".format(filename))
        
        
        img_color = cv2.imread(file_img_color)
        img_color = cv2.resize(img_color, shape_texture)
        if os.path.exists(file_img_rough):
            img_rough = cv2.imread(file_img_rough)
            img_rough = cv2.resize(img_rough, (img_color.shape[1], img_color.shape[0]))
        else:
            img_metal = np.zeros(img_color.shape)
        if os.path.exists(file_img_metal):
            img_metal = cv2.imread(file_img_metal)
            img_metal = cv2.resize(img_metal, (img_color.shape[1], img_color.shape[0]))
        else:
            img_metal = np.zeros(img_color.shape)
        
        if os.path.exists(file_img_normal):
            img_normal = cv2.imread(file_img_normal)
            img_normal = cv2.resize(img_normal, (img_color.shape[1], img_color.shape[0]))
        else:
            img_normal = np.zeros(img_color.shape)
        if os.path.exists(file_img_displacement):
            img_displacement = cv2.imread(file_img_displacement)
            img_displacement = cv2.resize(img_displacement, (img_color.shape[1], img_color.shape[0]))
        else:
            img_displacement = np.zeros(img_color.shape)
        
        imgs_color.append(img_color)
        imgs_rough.append(img_rough)
        imgs_metal.append(img_metal)
        imgs_normal.append(img_normal)
        imgs_displacement.append(img_displacement)
    
    if nb_texture>1:
        imgs_color, imgs_rough, imgs_displacement, imgs_metal, imgs_normal,  = concatenate_imgs(imgs_color=imgs_color,
                                                                                               imgs_rough=imgs_rough,
                                                                                               imgs_metal=imgs_metal,
                                                                                               imgs_normal=imgs_normal,
                                                                                               imgs_displacement=imgs_displacement)
    else:
        imgs_color = np.array(imgs_color[0])
        imgs_rough = np.array(imgs_rough[0])
        imgs_metal = np.array(imgs_metal[0])
        imgs_displacement = np.array(imgs_displacement[0])
        imgs_normal = np.array(imgs_normal[0])
                                                                                 
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        
    current_frame = bpy.data.scenes['Scene'].frame_current
    file_img_color = os.path.join(path_save, "temp_{}_color.jpg".format(current_frame))
    file_img_rough = os.path.join(path_save, "temp_{}_rough.jpg".format(current_frame))
    file_img_metal = os.path.join(path_save, "temp_{}_metal.jpg".format(current_frame))
    file_img_normal = os.path.join(path_save, "temp_{}_normal.jpg".format(current_frame))
    file_img_displacement = os.path.join(path_save, "temp_{}_displacement.jpg".format(current_frame))
    print(imgs_color.shape, file_img_color)
    cv2.imwrite(file_img_color, imgs_color)
    cv2.imwrite(file_img_rough, imgs_rough)
    cv2.imwrite(file_img_metal, imgs_metal)
    cv2.imwrite(file_img_normal, imgs_normal)
    cv2.imwrite(file_img_displacement, imgs_displacement)
    
        
    name_mat = "mat_{}".format(texture_index)
    mat = bpy.data.materials.new(name=name_mat)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    
    
    texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    texImage.image = bpy.data.images.load(file_img_color)
    mat.node_tree.links.new(bsdf.inputs['Base Color'],
                            texImage.outputs['Color'])
                            
    texCoord = mat.node_tree.nodes.new('ShaderNodeTexCoord')
    mat.node_tree.links.new(texImage.inputs['Vector'],
                            texCoord.outputs['Generated']) 
                                          
    
    texImage_rough = mat.node_tree.nodes.new('ShaderNodeTexImage')
    texImage_rough.image = bpy.data.images.load(file_img_rough)
    texImage_rough.image.colorspace_settings.name = 'Non-Color'
    mat.node_tree.links.new(bsdf.inputs['Roughness'],
                            texImage_rough.outputs['Color'])
    mat.node_tree.links.new(texImage_rough.inputs['Vector'],
                            texCoord.outputs['Generated']) 
                            
    
    
    if apply_normal_material:
        texImage_normal = mat.node_tree.nodes.new('ShaderNodeTexImage')
        texImage_normal.image = bpy.data.images.load(file_img_normal)
        texImage_normal.image.colorspace_settings.name = 'Non-Color'
                            
        texMapNormal = mat.node_tree.nodes.new('ShaderNodeNormalMap')
        mat.node_tree.links.new(texMapNormal.inputs['Color'],
                                texImage_normal.outputs['Color']) 
        mat.node_tree.links.new(texImage_normal.inputs['Vector'],
                                texCoord.outputs['Generated']) 
                            
    
    
    
        texImage_displacement = mat.node_tree.nodes.new('ShaderNodeTexImage')
        texImage_displacement.image = bpy.data.images.load(file_img_displacement)
        texImage_displacement.image.colorspace_settings.name = 'Non-Color' 
             
        texMapDisplacement = mat.node_tree.nodes.new('ShaderNodeDisplacement')
        mat.node_tree.links.new(texMapDisplacement.inputs['Height'],
                                texImage_displacement.outputs['Color']) 
        mat.node_tree.links.new(texImage_displacement.inputs['Vector'],
                                texCoord.outputs['Generated']) 
                            
        mat.node_tree.links.new(texMapDisplacement.inputs['Normal'],
                                texMapNormal.outputs['Normal'])                                  
                                       
        mat.node_tree.links.new(mat.node_tree.nodes['Material Output'].inputs['Displacement'],
                                texMapDisplacement.outputs["Displacement"])
    
                            
               
                    
                
    if os.path.exists(file_img_metal):
        texImage_metal = mat.node_tree.nodes.new('ShaderNodeTexImage')
        texImage_metal.image = bpy.data.images.load(file_img_metal)
        texImage_metal.image.colorspace_settings.name = 'Non-Color'
        mat.node_tree.links.new(bsdf.inputs['Metallic'],
                                texImage_metal.outputs['Color'])
        mat.node_tree.links.new(texImage_metal.inputs['Vector'],
                            texCoord.outputs['Generated']) 
                                
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
    
    

def apply_random_texture(nb_texture=25):
    for material in bpy.data.materials:
        material.user_clear()
        bpy.data.materials.remove(material)
        
    texture_index = 0
    for obj in bpy.data.objects:
        if obj.name!="Vide" and obj.name!="Camera" and obj.name!="Spot" and len(obj.children)==0 and obj.data is not None:
            
            mat, img_displacement = read_material(provider="ambientcg",
                                                      texture_index=texture_index,
                                                      nb_texture=nb_texture)
            
                
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
            



def coord_lamps_digi(rho):
    L = rho*np.array([[-0.06059872, -0.44839055, 0.8917812],
                    [-0.05939919, -0.33739538, 0.93948714],
                    [-0.05710194, -0.21230722, 0.97553319],
                    [-0.05360061, -0.07800089, 0.99551134],
                    [-0.04919816, 0.05869781, 0.99706274],
                    [-0.04399823, 0.19019233, 0.98076044],
                    [-0.03839991, 0.31049925, 0.9497977],
                    [-0.03280081, 0.41611025, 0.90872238],
                    [-0.18449839, -0.43989616, 0.87889232],
                    [-0.18870114, -0.32950199, 0.92510557],
                    [-0.1901994, -0.20549935, 0.95999698],
                    [-0.18849605, -0.07269848, 0.97937948],
                    [-0.18329657, 0.06229884, 0.98108166],
                    [-0.17500445, 0.19220488, 0.96562453],
                    [-0.16449474, 0.31129005, 0.93597008],
                    [-0.15270716, 0.4160195, 0.89644202],
                    [-0.30139786, -0.42509698, 0.85349393],
                    [-0.31020115, -0.31660118, 0.89640333],
                    [-0.31489186, -0.19549495, 0.92877599],
                    [-0.31450962, -0.06640203, 0.94692897],
                    [-0.30880699, 0.06470146, 0.94892147],
                    [-0.2981084, 0.19100538, 0.93522635],
                    [-0.28359251, 0.30729189, 0.90837601],
                    [-0.26670649, 0.41020998, 0.87212122],
                    [-0.40709586, -0.40559588, 0.81839168],
                    [-0.41919869, -0.29999906, 0.85689732],
                    [-0.42618633, -0.18329412, 0.88587159],
                    [-0.42691512, -0.05950211, 0.90233197],
                    [-0.42090385, 0.0659006, 0.90470827],
                    [-0.40860354, 0.18720162, 0.89330773],
                    [-0.39141794, 0.29941372, 0.87013988],
                    [-0.3707838, 0.39958255, 0.83836338],
                    [-0.499596, -0.38319693, 0.77689378],
                    [-0.51360334, -0.28130183, 0.81060526],
                    [-0.52190667, -0.16990217, 0.83591069],
                    [-0.52326874, -0.05249686, 0.85054918],
                    [-0.51720021, 0.06620003, 0.85330035],
                    [-0.50428312, 0.18139393, 0.84427174],
                    [-0.48561334, 0.28870793, 0.82512267],
                    [-0.46289771, 0.38549809, 0.79819605],
                    [-0.57853599, -0.35932235, 0.73224555],
                    [-0.59329349, -0.26189713, 0.76119165],
                    [-0.60202327, -0.15630604, 0.78303027],
                    [-0.6037003, -0.04570002, 0.7959004],
                    [-0.59781529, 0.06590169, 0.79892043],
                    [-0.58486953, 0.17439091, 0.79215873],
                    [-0.56588359, 0.27639198, 0.77677747],
                    [-0.54241965, 0.36921337, 0.75462733],
                    [0.05220076, -0.43870637, 0.89711304],
                    [0.05199786, -0.33138635, 0.9420612],
                    [0.05109826, -0.20999284, 0.97636672],
                    [0.04919919, -0.07869871, 0.99568366],
                    [0.04640163, 0.05630197, 0.99733494],
                    [0.04279892, 0.18779527, 0.98127529],
                    [0.03870043, 0.30950341, 0.95011048],
                    [0.03440055, 0.41730662, 0.90811441],
                    [0.17290651, -0.43181626, 0.88523333],
                    [0.17839998, -0.32509996, 0.92869988],
                    [0.18160174, -0.20480196, 0.96180921],
                    [0.18200745, -0.07490306, 0.98044012],
                    [0.17919505, 0.05849838, 0.98207285],
                    [0.17329685, 0.18839658, 0.96668244],
                    [0.1649036, 0.30880674, 0.93672045],
                    [0.1549931, 0.41578148, 0.89616009],
                    [0.28720483, -0.41910705, 0.8613145],
                    [0.29740177, -0.31410186, 0.90160535],
                    [0.30420604, -0.1965039, 0.9321185],
                    [0.30640529, -0.07010121, 0.94931639],
                    [0.30361153, 0.05950226, 0.95093613],
                    [0.29588748, 0.18589214, 0.93696036],
                    [0.28409783, 0.30349768, 0.90949304],
                    [0.26939905, 0.40849857, 0.87209694],
                    [0.39120402, -0.40190413, 0.8279085],
                    [0.40481085, -0.29960803, 0.86392315],
                    [0.41411685, -0.18590756, 0.89103626],
                    [0.41769724, -0.06449957, 0.906294],
                    [0.41498764, 0.05959822, 0.90787296],
                    [0.40607977, 0.18089099, 0.89575537],
                    [0.39179226, 0.29439419, 0.87168279],
                    [0.37379609, 0.39649585, 0.83849122],
                    [0.48278794, -0.38169046, 0.78818031],
                    [0.49848546, -0.28279175, 0.8194761],
                    [0.50918069, -0.1740934, 0.84286803],
                    [0.51360856, -0.05870098, 0.85601427],
                    [0.51097962, 0.05899765, 0.8575658],
                    [0.50151639, 0.17420569, 0.84742769],
                    [0.48600297, 0.28260173, 0.82700506],
                    [0.46600106, 0.38110087, 0.79850181],
                    [0.56150442, -0.35990283, 0.74510586],
                    [0.57807114, -0.26498677, 0.77176147],
                    [0.58933134, -0.1617086, 0.7915421],
                    [0.59407609, -0.05289787, 0.80266769],
                    [0.59157958, 0.057798, 0.80417224],
                    [0.58198189, 0.16649482, 0.79597523],
                    [0.56620006, 0.26940003, 0.77900008],
                    [0.54551481, 0.36380988, 0.7550205]], dtype=float)
    L1 =  np.zeros(L.shape)
    L1[:,0] = L[:,0]
    L1[:,1] = L[:,1]
    L1[:,2] = -(rho-L[:,2])
    return L1

def random_digi_rectangle(rho, nb_lamp=96):
    L = []
    for i in range(nb_lamp):
        x = np.random.uniform(-0.65,0.65)
        y = np.random.uniform(-0.55, 0.55)
        z = np.random.uniform(0.7, 1)
        L.append([x,y,z])
    L = rho*np.array(L)
    L[:,2] = -(rho-L[:,2])
    return L
    
    
    
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




def random_coord_lamp_sphere(samples, rayon=1, x_center=0, y_center=0, z_center=0, max_distance_object=4):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    while len(points)<samples:
        i = np.random.uniform(0,samples-1)
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        
        x = rayon*x+x_center
        y = rayon*y+y_center
        z = rayon*z+z_center
        
        if x>-max_distance_object:
            points.append((x, y, z))

    return np.array(points)  


def location(coordinates):

    x = [vertex.x for vertex in coordinates]
    y = [vertex.y for vertex in coordinates]
    z = [vertex.z for vertex in coordinates]

    return {"x": np.mean(x), "y": np.mean(y), "z": np.mean(z)}

    
def translation_obj(obj, x_change, y_change, z_change):
     obj.location[0]+=x_change
     obj.location[1]+=y_change
     obj.location[2]+=z_change
     
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
    obj.location[0]=0
    obj.location[1]=0
    obj.location[2]=-rho
    
            
    
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
            
            
def take_picture_view(obj, folder_name_save, path_dataset, coord_lamps,
 x_rotate, y_rotate, z_rotate, rho):
    
    current_frame = bpy.data.scenes['Scene'].frame_current
    depth = True
    
    nodetree = bpy.context.scene.node_tree
    
    nodetree.links.new(nodetree.nodes["Normaliser"].outputs["Value"],
                        nodetree.nodes['Sortie fichier.001'].inputs["Image"])
    nodetree.links.new(nodetree.nodes['Calques de rendu'].outputs['Normal'],
                        nodetree.nodes['Sortie fichier.002'].inputs["Image"])
    nodetree.links.new(nodetree.nodes['Calques de rendu'].outputs['AO'],
                        nodetree.nodes['Sortie fichier.003'].inputs["Image"])
    """
    nodetree.links.new(nodetree.nodes['Calques de rendu'].outputs['Shadow'],
                        nodetree.nodes['Sortie fichier.004'].inputs["Image"])
    """                      
    
                
    already_done = True
    for coord in coord_lamps:
        random_angle_add = np.random.rand(3)*0.2
                        
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
        bpy.data.lights['Spot'].energy = np.random.uniform(0.3,1.8)


        num_Image_output_render = str(current_frame)
        while(len(num_Image_output_render)<4):
            num_Image_output_render="0"+num_Image_output_render
        name_Image_output_render = "Image"+num_Image_output_render
        
        file_input = os.path.join(path_dataset, "img", "{}.{}".format(name_Image_output_render, format))
        folder = os.path.join(path_dataset, "img", folder_name_save, "view_{}_{}_{}".format(x_rotate, y_rotate, z_rotate))
        if not os.path.exists(folder):
            os.makedirs(folder)
            already_done = False
                            
        if not already_done:
            file_output = os.path.join(folder, "Image_{}_{}_{}_{}_{}.{}".format(x, y, z, rho, bpy.data.lights['Spot'].energy, format))
            if not os.path.exists(file_output):
                b=bpy.ops.render.render()
                
                
                if len(nodetree.nodes['Sortie fichier.001'].inputs["Image"].links)==1:
                    nodetree.links.remove(nodetree.nodes['Sortie fichier.001'].inputs["Image"].links[0])
                if len(nodetree.nodes['Sortie fichier.002'].inputs["Image"].links)==1:
                    nodetree.links.remove(nodetree.nodes['Sortie fichier.002'].inputs["Image"].links[0])
                if len(nodetree.nodes['Sortie fichier.003'].inputs["Image"].links)==1:
                    nodetree.links.remove(nodetree.nodes['Sortie fichier.003'].inputs["Image"].links[0])
                            
                #time.sleep(0.01)
                if os.path.exists(file_output):
                    os.remove(file_output)
                shutil.move(file_input, file_output)
                
                """
                file_shadow_input = os.path.join(path_dataset, "shadow", "{}.{}".format(name_Image_output_render, format))
                folder = os.path.join(path_save, "shadow", folder_name_save)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                file_shadow_output = os.path.join(folder, "Image_{}_{}_{}_{}.{}".format(x, y, z, rho, format))
                if os.path.exists(file_shadow_output):
                    os.remove(file_shadow_output)
                shutil.move(file_shadow_input, file_shadow_output) 
                """
                
                if depth:
                    file_scan_input = os.path.join(path_dataset, "depth", "{}.{}".format(name_Image_output_render, format))
                    folder = os.path.join(path_save, "depth", folder_name_save)
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                        
                    file_scan_output = os.path.join(folder, "view_{}_{}_{}.{}".format(x_rotate, y_rotate, z_rotate, format))
                    if os.path.exists(file_scan_output):
                        os.remove(file_scan_output)
                    shutil.move(file_scan_input, file_scan_output)
        
                    file_normal_input = os.path.join(path_dataset, "normal", "{}.{}".format(name_Image_output_render, format))
                    folder = os.path.join(path_save, "normal", folder_name_save)
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                        
                    file_normal_output = os.path.join(folder, "view_{}_{}_{}.{}".format(x_rotate, y_rotate, z_rotate, format))
                    if os.path.exists(file_normal_output):
                        os.remove(file_normal_output)
                        
                    shutil.move(file_normal_input, file_normal_output)
                    
                    
                    file_mask_input = os.path.join(path_dataset, "mask", "{}.{}".format(name_Image_output_render, format))
                    folder = os.path.join(path_save, "mask", folder_name_save)
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                        
                    file_mask_output = os.path.join(folder, "view_{}_{}_{}.{}".format(x_rotate, y_rotate, z_rotate, format))
                    if os.path.exists(file_mask_output):
                        os.remove(file_mask_output)
                        
                    shutil.move(file_mask_input, file_mask_output)
                               
                    depth = False
      
      


def load_object(file, folder_name_save, path_dataset, params):
    print("file", file)
    if ".gltf" in file:
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
    
    
    bpy.data.objects['Vide'].location[0] = 0
    bpy.data.objects['Vide'].location[1] = 0 
    bpy.data.objects['Vide'].location[2] = -params["rho"]
    
    for material in bpy.data.materials:
        material.user_clear()
        bpy.data.materials.remove(material)
    

def check_object_done(obj_name, x_rotate, y_rotate, z_rotate):
    folder_check = os.path.join(path_common_check, obj_name, "view_{}_{}_{}".format(x_rotate, y_rotate, z_rotate))
    
    if os.path.exists(folder_check):
        return True
    else:
        os.makedirs(folder_check)
        file = os.path.join(folder_check, name_process)
        open(file, 'a').close()
        time.sleep(0.05)
        if len(os.listdir(folder_check))==1:
            return False
        else:
            return True
        
    
         
def process_file(file, folder_name_save, path_dataset):
    for scene in bpy.data.scenes:
        for node in scene.node_tree.nodes:
            if node.type == 'OUTPUT_FILE':
                if "/img" in node.base_path:
                    node.base_path = os.path.join(path_save, "img")
                elif "/normal" in node.base_path:
                    node.base_path = os.path.join(path_save, "normal")
                elif "/depth" in node.base_path:
                    node.base_path = os.path.join(path_save, "depth")
                elif "/mask" in node.base_path:
                    node.base_path = os.path.join(path_save, "mask")
                
            
    params = {}
    params["rho"] = 1

    distance_max_camera_lamp = 1
    """
    coord_lamps = fibonacci_sphere(samples=200, rayon=params["rho"],
                                   x_center=-params["rho"])
    
    coord_lamps = coord_lamps_digi(rho=params["rho"])
    
    coord_lamps = random_coord_lamp_sphere(nb_image, x_center=0, y_center=0,
                                           z_center=-params["rho"], rayon=params["rho"],
                                            max_distance_object=distance_max_camera_lamp)
    coord_lamps = coord_lamps[np.abs(coord_lamps[:,2])<distance_max_camera_lamp]
    """
    coord_lamps = random_digi_rectangle(rho=params["rho"])
    #bpy.context.collection.objects.link(obj_save)
    process_one_view = False
    x_rotate = np.random.uniform(0, 2*np.pi)
    y_rotate = np.random.uniform(0, 2*np.pi)
    z_rotate = np.random.uniform(0, 2*np.pi)
    
    
    folder = os.path.join(path_dataset, "img", folder_name_save,
                        "view_{}_{}_{}".format(x_rotate, y_rotate, z_rotate))
    if not os.path.exists(folder) and not check_object_done(folder_name_save, x_rotate, y_rotate, z_rotate):
        load_object(file=file, folder_name_save=folder_name_save,
                    path_dataset=path_dataset,
                    params=params)
                
        obj = bpy.context.object
                
        obj.rotation_euler[0] = x_rotate
        obj.rotation_euler[1] = y_rotate
        obj.rotation_euler[2] = z_rotate
                
        camera = bpy.data.objects["Camera"]
        camera.location[0] = 0
        camera.location[1] = 0
        camera.location[2] = 0
        
           
        bpy.ops.object.select_all(action='DESELECT')
        mesh = [m for m in bpy.context.scene.objects if m.type == 'MESH']
        for obj in mesh:
            obj.select_set(state=True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.join()
                    

        apply_random_texture(nb_texture=nb_texture_per_object)

        take_picture_view(obj=obj, folder_name_save=folder_name_save,
                          path_dataset=path_dataset, coord_lamps=coord_lamps,
                          x_rotate=x_rotate, y_rotate=y_rotate, z_rotate=z_rotate,
                          rho=params["rho"])
                
        for object in bpy.data.objects:
            if object.name!="Vide" and object.name!="Camera" and object.name!="Spot":
                remove_object(object.name)
                
        check_object_done(folder_name_save, x_rotate, y_rotate, z_rotate)



def get_available_GPU():
    deviceIDs = GPUtil.getAvailable(limit = 100,
                                     maxLoad = 0.1,
                                     maxMemory = 0.1)
    for i in range(10):
        time.sleep(0.2)
        temp = GPUtil.getAvailable(limit = 100,
                                   maxLoad = 0.1,
                                   maxMemory = 0.1)
        for device in deviceIDs:
            if device not in temp:
                deviceIDs.remove(device)
                
    return deviceIDs


def select_available_GPU(nb_max_GPU=1, id_use_gpu=None):
    if id_use_gpu is None:
        available_GPU = get_available_GPU()            
    deviceList = bpy.context.preferences.addons["cycles"].preferences.get_devices()
    idx = 0
    use_gpu=0
    for deviceTuple in deviceList:
        for device in deviceTuple:
            if (device.type=="CUDA" and id_use_gpu is not None) or (device.type=="CUDA" and idx in available_GPU and use_gpu<nb_max_GPU):
                if id_use_gpu is None or idx==id_use_gpu:
                    device.use = True
                    use_gpu+=1
                    
                else:
                    device.use = False
            else:
                device.use = False
            idx+=1


def start_object():
    print("Begin process object")

    path_object = os.path.join(path_obj, "object")
    folder_object = np.random.choice(os.listdir(path_object), 1)[0]
    print(folder_object)
    file= os.path.join(path_object, folder_object)
    if os.path.isdir(file):
        file = os.path.join(file, "scene.gltf")
        
    process_file(file=file,
                 folder_name_save=folder_object,
                 path_dataset=path_save)

            
def start_blobby_david():
    print("Begin process blobby David")
    path_blobby_david = os.path.join(path_obj, "blobby")
    filename = np.random.choice(os.listdir(path_blobby_david), 1)[0]
    if ".mtl" not in filename:
        file = os.path.join(path_blobby_david, filename)
        process_file(file=file,
                     folder_name_save=filename.split('.obj')[0],
                     path_dataset=path_save)
   

def start_shape_NETCore():
    print("Begin process shape NetCore")
    path_shapeNetCore = os.path.join(path_obj, "shapeNetCore")
    for folder_object in np.random.choice(os.listdir(path_shapeNetCore), 20):
        file = os.path.join(path_shapeNetCore, folder_object, "model.obj")
        process_file(file=file,
                     folder_name_save=folder_object,
                     path_dataset=path_save)

def start_shape_Net_Sem():
    print("Begin process shape NetSem")
    path_shapeNetSem = os.path.join(path_obj, "shapeNetSem")
    for filename in np.random.choice(os.listdir(path_shapeNetSem), 20):
        if ".mtl" not in filename:
            file = os.path.join(path_shapeNetSem, filename)
            process_file(file=file,
                         folder_name_save=filename.split('.obj')[0],
                         path_dataset=path_save)

def start_ABC():
    print("Begin process ABC")
    path_ABC = os.path.join(path_obj, "ABC")
    for filename in np.random.choice(os.listdir(path_ABC), 50):
        if ".mtl" not in filename:
            file = os.path.join(path_ABC, filename)
            process_file(file=file,
                         folder_name_save=filename.split('.obj')[0],
                         path_dataset=path_save)


def start_Thingi10K():
    print("Begin process Thingi10K")
    path_Thingi10k = os.path.join(path_obj, "Thingi10K")
    for filename in np.random.choice(os.listdir(path_Thingi10k), 20):
        file = os.path.join(path, filename)
        process_file(file=file,
                    folder_name_save=filename.split('.stl')[0],
                    path_dataset=path_save)

def start_blob_Clement():
    print("Begin process blob_Clement")
    path_blob_Clement = os.path.join(path_obj, "blobby_clement")
    for filename in np.random.choice(os.listdir(path_blob_Clement), 1):
        file = os.path.join(path_blob_Clement, filename)
        process_file(file=file,
                    folder_name_save=filename.split('.obj')[0],
                    path_dataset=path_save)  

















max_ram_use = 70 + np.random.randint(-10,10)
pid = os.getpid()


def kill_too_much_ram():
    if psutil.virtual_memory().percent>max_ram_use:
        print("percent use", psutil.virtual_memory().percent)
        #os.kill(pid, signal.SIGSTOP)
        command = "kill {}".format(pid)
        subprocess.call(command, shell=True)
        
sched = BackgroundScheduler(timezone="Europe/Paris")
sched.add_job(kill_too_much_ram, 'interval',
              seconds=0.5)
sched.start()       

        
name_process = uuid.uuid4().hex
path = "/data/chercheurs/hardy216/data/blender_data_generation"
path_common_check = "/home/hardy216/Desktop/test/"
path_common_check = "/home/personnels/hardy216/code/object_done/"
#path = "/home/hardy216/Desktop/"
path_save = os.path.join(path, "blender_result")
path_obj = os.path.join(path, "blender_obj")

path_texture = os.path.join(path_obj, "texture")
path_texture_texturescom = os.path.join(path_texture, "TexturesCom")
path_texture_ambientcg = os.path.join(path_texture, "ambientcg")

possible_features_ambientcg = os.listdir(path_texture_ambientcg)


bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value[2] = 0
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value[1] = 0
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value[0] = 0

bpy.context.scene.cycles.samples = 256

resolution = 128
nb_image = 96*2.1
nb_view = 3
nb_texture_per_object = 1
apply_normal_material = True


bpy.data.cameras['Camera'].type = "ORTHO"
bpy.context.scene.render.resolution_x = resolution
bpy.context.scene.render.resolution_y = resolution
bpy.data.cameras['Camera'].lens = 350
bpy.data.cameras['Camera'].ortho_scale = 0.7
bpy.data.objects['Vide'].location[0] = 0
bpy.data.objects['Vide'].location[1] = 0
bpy.data.objects['Vide'].location[2] = -1



gpu = None
num_process = None
for arg in sys.argv:
    if "gpu" in arg and "None" not in arg:
        gpu = int(arg.split("gpu=")[1])
    if "num" in arg and "None" not in arg:
        num_process = int(arg.split("num=")[1])

if num_process is None:
    num_process = 1  
    
     
format = "exr"
shape_texture = (1024, 1024)
nb_max_GPU = 1        
print("gpu", gpu)
select_available_GPU(nb_max_GPU=nb_max_GPU,
                     id_use_gpu=gpu)
if gpu is not None:
    currentFrame = bpy.data.scenes['Scene'].frame_current = int(gpu) + 100*num_process
else:
    currentFrame = bpy.data.scenes['Scene'].frame_current = 1
start_object()
print("finish")
