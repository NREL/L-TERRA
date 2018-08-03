# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:09:56 2015

@author: jnewman
"""

#This script reads in data from a lidar (in .npz format) and applies user-defined options in L-TERRA to correct the lidar TI.  
#After all TI data have been corrected, the timestamp, original TI, and corrected TI are written out to a CSV file. 

import glob
import numpy as np
from lidar_all_processing import lidar_processing_noise,lidar_processing_vol_averaging
import csv
from datetime import date
from itertools import compress

##########################################################

#USER INPUTS GO HERE

#Directory where CSV output file will be stored
main_directory = '/Users/jnewman/Desktop/Scripts/L_TERRA_Github/'

#Directory where lidar data are saved
lidar_directory = main_directory + 'Processed_ZephIR_Data/'

#Height were TI values will be corrected
height_needed = 60

#Time period for data extraction 
years = [2013]
months = [11]
days = np.arange(20,28)

#Wind speed minimum and maximum thresholds. If threshold is not needed, put 'none'.
U_min = 4
U_max = 'none'

#Wind direction sectors to exclude. If exclusion sector is not needed, put 'none' for wd_min and wd_max values.
wd_sector_min1 = 270
wd_sector_max1 = 360

wd_sector_min2 = 0
wd_sector_max2 = 90

#Model options to use for different stability conditions. If correction is not needed, put 'none'.

#Options for stable conditions (p >= 0.2)
mode_noise_s = 'spike'
mode_vol_s = 'acf'


#Options for neutral conditions (0.1 <= p < 0.2)
mode_noise_n = 'none'
mode_vol_n = 'spectral_correction'


#Options for unstable conditions (p < 0.1)
mode_noise_u = 'lenschow_spectrum'
mode_vol_u = 'none'

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

files_ZephIR = []

for i in years:
    for j in months:
        for k in days:
            #Find all lidar files that correspond to this date
            files_found_ZephIR = glob.glob(lidar_directory + '*' + str(i) + str(j).zfill(2) + str(k).zfill(2) + '*')
            for ii in files_found_ZephIR:
                files_ZephIR.append(ii)

TI_ZephIR_orig_all = np.zeros(len(files_ZephIR))
TI_ZephIR_orig_all[:] = np.nan

time_all = []

#Initialize TI arrays with NaNs. Only fill in data where wind direction and wind speed meet particular thresholds.
#For example, you may want to exclude wind direction angles affected by tower or turbine waking or wind speeds that would
#not be used in a power performance test. In this case, all measurements come from the lidar data. 
for i in range(len(files_ZephIR)):
    wd = np.load(files_ZephIR[i])['wd']
    U = np.load(files_ZephIR[i])['U']
    time_all.append(np.load(files_ZephIR[i])['time'].item())
    if ~(wd >= wd_sector_min1 and wd < wd_sector_max1) and ~(wd >= wd_sector_min2 and wd < wd_sector_max2) and U >=U_min and U < U_max:
        TI_ZephIR_orig_all[i] = (np.sqrt(np.load(files_ZephIR[i])['u_var'])/np.load(files_ZephIR[i])['U'])*100
          
TI_ZephIR_orig_all = np.array(TI_ZephIR_orig_all)

#Convert time from datetime format to a normal timestamp. 
#Timestamp is in UTC and corresponds to start of 10-min. averaging period.
timestamp_10min_all = []

for i in time_all:
    timestamp_10min_all.append(date.strftime(i,"%Y/%m/%d %H:%M"))
       
with open(main_directory + 'L_TERRA_corrected_TI_ZephIR.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    data = [['Timestamp (UTC)','Original TI (%)','Corrected TI (%)']]
    a.writerows(data)    

#Initialize arrays for the new lidar TI after each correction has been applied 
TI_ZephIR_noise_all = np.zeros(len(files_ZephIR))
TI_ZephIR_noise_all[:] = np.nan
TI_ZephIR_vol_avg_all = np.zeros(len(files_ZephIR))
TI_ZephIR_vol_avg_all[:] = np.nan

p_all = np.zeros(len(files_ZephIR))
p_all[:] = np.nan    

for i in range(len(files_ZephIR)):
    if ~np.isnan(TI_ZephIR_orig_all[i]):
        p_all[i] = np.load(files_ZephIR[i])['p']
        mode_ws = 'raw_ZephIR'
        frequency = 1./15
        file_temp = files_ZephIR[i]
        if p_all[i] >= 0.2:
            mode_noise = mode_noise_s
            mode_vol = mode_vol_s
        elif p_all[i] >= 0.1:
            mode_noise = mode_noise_n
            mode_vol = mode_vol_n
        else:
            mode_noise = mode_noise_u
            mode_vol = mode_vol_u        

        
        u_rot = np.load(file_temp)['u_rot']
        u_var = np.load(file_temp)['u_var']
        U = np.load(file_temp)['U']
                                          
        
        wd = np.load(files_ZephIR[i])['wd']
    
        #Apply noise correction and calculate variance
        if "none" in mode_noise:
            u_var_noise = u_var
        else:
            u_var_noise = lidar_processing_noise(u_rot,frequency,mode_ws,mode_noise)

        TI_ZephIR_noise_all[i] = (np.sqrt(u_var_noise)/U)*100
        
        #Estimate loss of variance due to volume averaging 
        if "none" in mode_vol:
            u_var_diff = 0.
        else:
            try:
            
                u_var_diff = lidar_processing_vol_averaging(u_rot,frequency,mode_ws,mode_vol)
                
            except:
                u_var_diff = 0.
         
        u_var_vol = u_var_noise + u_var_diff
        TI_ZephIR_vol_avg_all[i] = (np.sqrt(u_var_vol)/U)*100

#Extract TI values and timestamps for all times when corrected TI value is valid
mask = ~np.isnan(TI_ZephIR_vol_avg_all)

timestamp_10min_all = list(compress(timestamp_10min_all,mask))
TI_ZephIR_orig_all = TI_ZephIR_orig_all[mask]
TI_ZephIR_vol_avg_all = TI_ZephIR_vol_avg_all[mask]


#Reduce number of decimal places in output TI data                        
TI_orig_temp = ["%0.2f" % i for i in TI_ZephIR_orig_all]    
TI_corrected_temp = ["%0.2f" % i for i in TI_ZephIR_vol_avg_all]  
                           
#Write out timestamp, original lidar TI, and corrected lidar TI                      
with open(main_directory + 'L_TERRA_corrected_TI_ZephIR.csv', 'a') as fp:
     a = csv.writer(fp, delimiter=',')
     data = np.vstack([timestamp_10min_all,TI_orig_temp,TI_corrected_temp]).transpose()
     a.writerows(data)  



