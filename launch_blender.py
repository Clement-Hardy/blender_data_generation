#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 01:39:49 2021

@author: hardy216
"""
import subprocess
import time
import argparse
from tqdm import tqdm

path_to_blender = "/data/chercheurs/hardy216/data/blender_data_generation/blender-2.93.6/blender"
command = os.path.join(path_to_blender, "automatic_database_generation_serveur_greyc.blend")


parser = argparse.ArgumentParser()
parser.add_argument('-gpu', '--gpu', metavar='gpu',
                    required=False, help='gpu to use')
parser.add_argument('-time', '--time', metavar='gpu',
                    required=False, help='gpu to use')
parser.add_argument('-num', '--num', metavar='gpu',
                    required=False, help='gpu to use')

args = parser.parse_args()
if args.gpu is None:
    gpu = None
else:
    gpu = int(args.gpu)

if args.time is None:
    max_time = None
else:
    max_time = float(args.time)
    
if args.num is None:
    num_process = None
else:
    num_process = int(args.num)
    
print("gpu ", gpu)
print("max duration", max_time)
print("num process", num_process)
command = command+" --background --python pipeline_serveur.py -- gpu={} --num={}".format(gpu, num_process)

beginning = time.time()
for i in tqdm(range(3000)):
    if max_time is not None and (time.time()-beginning)/3600>max_time:
        break;
    subprocess.call(command, shell=True)
    
