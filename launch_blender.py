#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 01:39:49 2021

@author: hardy216
"""

import subprocess
import time


command = "./blender-2.93.6/blender automatic_database_generation_serveur_greyc.blend --background --python pipeline_serveur.py"
for i in range(100):
    subprocess.call(command, shell=True)
    time.sleep(10)