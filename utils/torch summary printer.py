# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 18:22:13 2021

@author: yipji
"""


import contextlib
from torchsummary import summary
 
file_path = 'randomfile.txt'
with open(file_path, "w") as o:
    with contextlib.redirect_stdout(o):
        summary(net, (3,224,224))