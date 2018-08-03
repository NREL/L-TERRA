# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:09:56 2015

@author: jnewman
"""

#This script reads in high-resolution ZephIR files (.ZPH), performs some basic calculations, and writes the output from each 
#10-min. period to a numpy file (.npz).

import glob
import numpy as np

from lidar_all_processing import ZephIR_processing_standard
from lidar_preprocessing_functions import get_10min_var

#Directory where .ZPH files are found
dir_ZPH = '/Users/jnewman/Desktop/Scripts/L_TERRA/Raw ZephIR Data/'

#Directory where .npz files should be saved 
dir_npz = '/Users/jnewman/Desktop/Scripts/L_TERRA_Github/Processed_ZephIR_Data/'
cmd = "mkdir -p " + dir_npz
import os
os.system(cmd)

#Height where variance and wind speed should be extracted
height_needed = 80


#Specify the time period for data extraction 
years = [2013]
months = [11]
days = np.arange(20,28)


for i in years:
    for j in months:
        for k in days:
            filenames_ZephIR = glob.glob(dir_ZPH + '*' + str(i) + '_M' + str(j).zfill(2) + '_D' + str(k).zfill(2) + '*')
            
            if len(filenames_ZephIR) != 0:
                for jj in filenames_ZephIR:    
                    
                    #Extract raw rotated wind speed, 10-min. mean wind speed, 10-min. mean wind direction, 10-min. shear parameter, 
                    #and 10-min. timestamp 
                    [u_rot,U,wd,p,time_datenum_10min] = ZephIR_processing_standard(jj,height_needed) 
                    #Calculate 10-min. streamwise variance
                    u_var = get_10min_var(u_rot,1./15)
                                                                 
                    #Write output to 10-min. files        
                    for kk in range(len(time_datenum_10min)):
                        
                        filename = dir_npz + "ZephIR_" + str(i) + str(j).zfill(2) + str(k).zfill(2) + "_" + \
                        str(time_datenum_10min[kk].hour).zfill(2) + str(time_datenum_10min[kk].minute).zfill(2) + "_UTC"
                        np.savez(filename,u_rot=u_rot[kk*40:(kk+1)*40 + 1],time=time_datenum_10min[kk],\
                        U = U[kk],wd=wd[kk],u_var=u_var[kk],p=p[kk])
                        
