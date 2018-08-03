# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:09:56 2015

@author: jnewman
"""

#This script reads in data from a lidar (in .npz format) and applies user-defined options in L-TERRA to correct the lidar TI.  
#After all TI data have been corrected, the timestamp, original TI, and corrected TI are written out to a CSV file. 

import glob
import numpy as np
from lidar_all_processing import lidar_processing_noise,lidar_processing_vol_averaging,lidar_processing_var_contam
import csv
from datetime import date
from itertools import compress

##########################################################

#USER INPUTS GO HERE

#Directory where CSV output file will be stored
main_directory = '/Users/jnewman/Desktop/Scripts/L_TERRA_Github/'

#Directory where lidar data are saved
lidar_directory = main_directory + 'Processed_WC_Data/'

#Height were TI values will be corrected
height_needed = 60

#Time period for data extraction 
years = [2012]
months = [11]
days = np.arange(14,21)

#Wind speed minimum and maximum thresholds. If threshold is not needed, put 'none'.
U_min = 4
U_max = 'none'

#Wind direction sectors to exclude. If exclusion sector is not needed, put 'none' for wd_min and wd_max values.
wd_sector_min1 = 45
wd_sector_max1 = 135

wd_sector_min2 = 'none'
wd_sector_max2 = 'none'

#Model options to use for different stability conditions. If correction is not needed, put 'none'.

#Options for stable conditions (p >= 0.2)
mode_ws_s = 'raw_WC'
mode_noise_s = 'spike'
mode_vol_s = 'acf'
mode_contamination_s = 'taylor_ws'

#Options for neutral conditions (0.1 <= p < 0.2)
mode_ws_n = 'raw_WC'
mode_noise_n = 'none'
mode_vol_n = 'none'
mode_contamination_n = 'taylor_var'

#Options for unstable conditions (p < 0.1)
mode_ws_u = 'VAD'
mode_noise_u = 'spike'
mode_vol_u = 'none'
mode_contamination_u = 'taylor_ws'



##########################################################

if 'none' in str(U_min):
    U_min = 0
if 'none' in str(U_max):
    U_max = np.inf   
if 'none' in str(wd_sector_min1):
    wd_sector_min1 = np.nan
if 'none' in str(wd_sector_max1):
    wd_sector_max1 = np.nan   
if 'none' in str(wd_sector_min2):
    wd_sector_min2 = np.nan
if 'none' in str(wd_sector_max2):
    wd_sector_max2 = np.nan       

files_WC_DBS = []
files_WC_VAD = []
files_WC_vr = []


for i in years:
    for j in months:
        for k in days:
            #Find all lidar files that correspond to this date
            files_found_WC = glob.glob(lidar_directory + '*' + str(i) + str(j).zfill(2) + str(k).zfill(2) + '*')
            for jj in files_found_WC:
                if "DBS" in jj:
                    files_WC_DBS.append(jj)
                if "VAD" in jj:
                    files_WC_VAD.append(jj)
                if "vr" in jj:
                    files_WC_vr.append(jj)


TI_WC_orig_all = np.zeros(len(files_WC_DBS))
TI_WC_orig_all[:] = np.nan

time_all = []

#Initialize TI arrays with NaNs. Only fill in data where wind direction and wind speed meet particular thresholds.
#For example, you may want to exclude wind direction angles affected by tower or turbine waking or wind speeds that would
#not be used in a power performance test. In this case, all measurements come from the lidar data. 
for i in range(len(files_WC_DBS)):
    wd = np.load(files_WC_DBS[i])['wd']
    U = np.load(files_WC_DBS[i])['U']
    time_all.append(np.load(files_WC_DBS[i])['time'].item())
    if ~(wd >= wd_sector_min1 and wd < wd_sector_max1) and ~(wd >= wd_sector_min2 and wd < wd_sector_max2) and U >=U_min and U < U_max:
        TI_WC_orig_all[i] = (np.sqrt(np.load(files_WC_DBS[i])['u_var'])/np.load(files_WC_DBS[i])['U'])*100
          
TI_WC_orig_all = np.array(TI_WC_orig_all)

#Convert time from datetime format to a normal timestamp. 
#Timestamp is in UTC and corresponds to start of 10-min. averaging period.
timestamp_10min_all = []

for i in time_all:
    timestamp_10min_all.append(date.strftime(i,"%Y/%m/%d %H:%M"))
       
with open(main_directory + 'L_TERRA_corrected_TI_WC.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    data = [['Timestamp (UTC)','Original TI (%)','Corrected TI (%)']]
    a.writerows(data)    

#Initialize arrays for the new lidar TI after each correction has been applied 
TI_WC_noise_all = np.zeros(len(files_WC_DBS))
TI_WC_noise_all[:] = np.nan
TI_WC_vol_avg_all = np.zeros(len(files_WC_DBS))
TI_WC_vol_avg_all[:] = np.nan
TI_WC_var_contamination_all = np.zeros(len(files_WC_DBS))
TI_WC_var_contamination_all[:] = np.nan

p_all = np.zeros(len(files_WC_DBS))
p_all[:] = np.nan    

for i in range(len(files_WC_DBS)):
    if ~np.isnan(TI_WC_orig_all[i]):
        p_all[i] = np.load(files_WC_DBS[i])['p']
        if p_all[i] >= 0.2:
            mode_ws = mode_ws_s
            mode_noise = mode_noise_s
            mode_vol = mode_vol_s
            mode_contamination = mode_contamination_s
        elif p_all[i] >= 0.1:
            mode_ws = mode_ws_n
            mode_noise = mode_noise_n
            mode_vol = mode_vol_n
            mode_contamination = mode_contamination_n
        else:
            mode_ws = mode_ws_u
            mode_noise = mode_noise_u
            mode_vol = mode_vol_u
            mode_contamination = mode_contamination_u
            
        if "raw" in mode_ws:
            frequency = 1.
            file_temp = files_WC_DBS[i]
        else:
            frequency = 1./4
            file_temp = files_WC_VAD[i]
        
        u_rot = np.load(file_temp)['u_rot']
        u_var = np.load(file_temp)['u_var']
        U = np.load(file_temp)['U']
                                          
        vr_n = np.load(files_WC_vr[i])['vr_n']
        vr_e = np.load(files_WC_vr[i])['vr_e']
        vr_s = np.load(files_WC_vr[i])['vr_s']
        vr_w = np.load(files_WC_vr[i])['vr_w']
        vert_beam = np.load(files_WC_VAD[i])['w_rot']
        
        wd = np.load(files_WC_DBS[i])['wd']
    
        #Apply noise correction and calculate variance
        if "none" in mode_noise:
            u_var_noise = u_var
        else:
            u_var_noise = lidar_processing_noise(u_rot,frequency,mode_ws,mode_noise)

        TI_WC_noise_all[i] = (np.sqrt(u_var_noise)/U)*100
        
        #Estimate loss of variance due to volume averaging 
        if "none" in mode_vol:
            u_var_diff = 0.
        else:
            try:
            
                u_var_diff = lidar_processing_vol_averaging(u_rot,frequency,mode_ws,mode_vol)
                
            except:
                u_var_diff = 0.
         
        u_var_vol = u_var_noise + u_var_diff
        TI_WC_vol_avg_all[i] = (np.sqrt(u_var_vol)/U)*100
        
        #Estimate increase in variance due to variance contamination 
        if "none" in mode_contamination:
            u_var_diff = 0.
        else:
            u_var_diff = lidar_processing_var_contam(vr_n,vr_e,vr_s,vr_w,vert_beam,wd,U,height_needed,1./4,62.,mode_contamination)
            
            if np.isnan(u_var_diff):
                    u_var_diff = 0.
 
        
        u_var_contam = u_var_vol - u_var_diff
        
        TI_WC_var_contamination_all[i] = (np.sqrt(u_var_contam)/U)*100
            

#Extract TI values and timestamps for all times when corrected TI value is valid
mask = ~np.isnan(TI_WC_var_contamination_all)

timestamp_10min_all = list(compress(timestamp_10min_all,mask))
TI_WC_orig_all = TI_WC_orig_all[mask]
TI_WC_var_contamination_all = TI_WC_var_contamination_all[mask]


#Reduce number of decimal places in output TI data                        
TI_orig_temp = ["%0.2f" % i for i in TI_WC_orig_all]    
TI_corrected_temp = ["%0.2f" % i for i in TI_WC_var_contamination_all]  
                           
#Write out timestamp, original lidar TI, and corrected lidar TI                      
with open(main_directory + 'L_TERRA_corrected_TI_WC.csv', 'a') as fp:
     a = csv.writer(fp, delimiter=',')
     data = np.vstack([timestamp_10min_all,TI_orig_temp,TI_corrected_temp]).transpose()
     a.writerows(data)  



