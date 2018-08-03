# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:09:56 2015

@author: jnewman
"""

#This script reads in 1 Hz WINDCUBE files (.rtd), performs some basic calculations, and writes the output from each 
#10-min. period to a numpy file (.npz).

import glob
import numpy as np

from lidar_all_processing import WC_processing_standard
from lidar_preprocessing_functions import get_10min_var

#Directory where .rtd files are found
dir_rtd = '/Users/jnewman/Desktop/Scripts/L_TERRA/Raw WC Data/'

#Directory where .npz files should be saved 
dir_npz = '/Users/jnewman/Desktop/Scripts/L_TERRA_Github/Processed_WC_Data/'
cmd = "mkdir -p " + dir_npz
import os
os.system(cmd)

#Height where variance and wind speed should be extracted
height_needed = 60


#Specify the time period for data extraction 
years = [2012]
months = [11]
days = np.arange(14,21)

for i in years:
    for j in months:
        for k in days:
            filenames_WC = glob.glob(dir_rtd + '*' + str(i) + '_' + str(j).zfill(2) + '_' + str(k).zfill(2) + '*')
            if len(filenames_WC) != 0:
                for jj in filenames_WC:   
                    #Extract raw rotated wind speed, 10-min. mean wind speed, 10-min. mean wind direction, 10-min. shear parameter, 
                    #and 10-min. timestamp using DBS technique. (DBS is the raw output from the .rtd files.)
                    [u_rot_DBS,U_DBS,wd_DBS,p,time_datenum_10min] = WC_processing_standard(jj,"raw",height_needed) 
                    #Calculate 10-min. streamwise variance
                    u_var_DBS = get_10min_var(u_rot_DBS,1.)
                    #Extract rotated streamwise wind speed from the VAD technique, 10-min. mean wind speed, and rotated vertical
                    #wind speed
                    [u_rot_VAD,U_VAD,w_rot] = WC_processing_standard(jj,"VAD",height_needed)
                    #Calculate 10-min. streamwise and vertical variance
                    u_var_VAD = get_10min_var(u_rot_VAD,1./4)
                    w_var = get_10min_var(w_rot,1./4)
                    
                    #Extract raw off-vertical radial wind speed components
                    [vr_n,vr_e,vr_s,vr_w] = WC_processing_standard(jj,"vr",height_needed) 

                    #Write output to 10-min. files 
                    for kk in range(len(time_datenum_10min)):
                        
                        filename = dir_npz + "WC_DBS_" + str(i) + str(j).zfill(2) + str(k).zfill(2) + "_" + \
                        str(time_datenum_10min[kk].hour).zfill(2) + str(time_datenum_10min[kk].minute).zfill(2) + "_UTC"
                        np.savez(filename,u_rot=u_rot_DBS[kk*600:(kk+1)*600 + 1],time=time_datenum_10min[kk],\
                        U = U_DBS[kk],wd=wd_DBS[kk],u_var=u_var_DBS[kk],p=p[kk])
                        
                        filename = dir_npz + "WC_VAD_" + str(i) + str(j).zfill(2) + str(k).zfill(2) + "_" + \
                        str(time_datenum_10min[kk].hour).zfill(2) + str(time_datenum_10min[kk].minute).zfill(2) + "_UTC"
                        np.savez(filename,u_rot=u_rot_VAD[kk*150:(kk+1)*150 + 1],time=time_datenum_10min[kk],\
                        U = U_VAD[kk],u_var=u_var_VAD[kk],vert_beam_var=w_var[kk],w_rot = w_rot[kk*150:(kk+1)*150 + 1])
                        
                        filename = dir_npz + "WC_vr_" + str(i) + str(j).zfill(2) + str(k).zfill(2) + "_" + \
                        str(time_datenum_10min[kk].hour).zfill(2) + str(time_datenum_10min[kk].minute).zfill(2) + "_UTC"
                        np.savez(filename,vr_n=vr_n[kk*150:(kk+1)*150 + 1],vr_e=vr_e[kk*150:(kk+1)*150 + 1],\
                        vr_s=vr_s[kk*150:(kk+1)*150 + 1],vr_w=vr_w[kk*150:(kk+1)*150 + 1],time=time_datenum_10min[kk])
                        
                        
             
                        
                        

