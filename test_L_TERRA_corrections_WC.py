# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:09:56 2015

@author: jnewman
"""

#This script reads in data from a WINDCUBE lidar and a reference instrument (in .npz format) and loops through all 
#possible combinations of corrections in L-TERRA. The TI mean absolute error (MAE) is calculated for each model combination
#and written to a CSV file, with MAE calculated separately for different stability conditions. After all model combinations have 
#been tested, the L-TERRA combinations that minimized the overall MAE, stable conditions MAE, neutral conditions MAE, and 
#unstable conditions MAE are written to the CSV file. 

import glob
import numpy as np
from lidar_all_processing import lidar_processing_noise,lidar_processing_vol_averaging,lidar_processing_var_contam
import csv
from sklearn.metrics import mean_absolute_error as MAE


MAE_min = 10
MAE_min_s = 10
MAE_min_n = 10
MAE_min_u = 10
opts_best = "None"
opts_best_s = "None"
opts_best_n = "None"
opts_best_u = "None"

##########################################################

#USER INPUTS GO HERE

#Directory where CSV output file will be stored
main_directory = '/Users/jnewman/Desktop/Scripts/L_TERRA_Github/'

#Directory where lidar data are saved
lidar_directory = main_directory + 'Processed_WC_Data/'

#Directory where data from reference device is saved 
reference_directory = main_directory + 'Processed Reference Data/'

#Height were TI values will be compared 
height_needed = 60

#Time period for data extraction 
years = [2012]
months = [11]
days = np.arange(14,21)

#Wind speed minimum and maximum thresholds. If threshold is not needed, put 'none'.
U_min = 4
U_max = 'none'

#Measurement to use for wind speed threshold ('reference' or 'lidar')
U_opt = 'reference'

#Wind direction sectors to exclude. If exclusion sector is not needed, put 'none' for wd_min and wd_max values.
wd_sector_min1 = 45
wd_sector_max1 = 135

wd_sector_min2 = 'none'
wd_sector_max2 = 'none'

#Measurement to use for wind direction sector ('reference' or 'lidar')
wd_opt = 'reference'

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


files_ref = []
files_WC_DBS = []
files_WC_VAD = []
files_WC_vr = []


for i in years:
    for j in months:
        for k in days:
            #Find all reference and lidar files that correspond to this date
            files_found_ref = glob.glob(reference_directory +'/*' + str(i) + str(j).zfill(2) + str(k).zfill(2) + '*')
            files_found_WC = glob.glob(lidar_directory + '*' + str(i) + str(j).zfill(2) + str(k).zfill(2) + '*')
            
            #Find 10-min. periods where data files for both the lidar and reference device exist 
            for ii in files_found_ref:
                str_match = str(ii[-21:])
                matching_WC = [s for s in files_found_WC if str_match in s]
                if(matching_WC):
                    files_ref.append(ii)
                    for jj in matching_WC:
                        if "DBS" in jj:
                            files_WC_DBS.append(jj)
                        if "VAD" in jj:
                            files_WC_VAD.append(jj)
                        if "vr" in jj:
                            files_WC_vr.append(jj)

                        
TI_ref_all = np.zeros(len(files_ref))
TI_ref_all[:] = np.nan

TI_WC_orig_all = np.zeros(len(files_ref))
TI_WC_orig_all[:] = np.nan

#Initialize TI arrays with NaNs. Only fill in data where wind direction and wind speed meet particular thresholds.
#For example, you may want to exclude wind direction angles affected by tower or turbine waking or wind speeds that would
#not be used in a power performance test. 
for i in range(len(files_ref)):
    if 'reference' in U_opt:
        file_ws = files_ref[i]
    else:
        file_ws = files_WC_DBS[i]
    if 'reference' in wd_opt:
        file_wd = files_ref[i]
    else:
        file_wd = files_WC_DBS[i]  
    U = np.load(file_ws)['U']
    wd = np.load(file_wd)['wd']
    if ~(wd >= wd_sector_min1 and wd < wd_sector_max1) and ~(wd >= wd_sector_min2 and wd < wd_sector_max2) and U >=U_min and U < U_max:
        TI_ref_all[i] = (np.sqrt(np.load(files_ref[i])['u_var'])/np.load(files_ref[i])['U'])*100
        TI_WC_orig_all[i] = (np.sqrt(np.load(files_WC_DBS[i])['u_var'])/np.load(files_WC_DBS[i])['U'])*100
    
     
with open(main_directory + 'L_TERRA_combination_summary_WC.csv', 'w') as fp:
    a = csv.writer(fp, delimiter=',')
    data = [['Module Options','MAE Overall','MAE Stable','MAE Neutral','MAE Unstable']]
    a.writerows(data)        


#Options to test in L-TERRA
ws_opts = ["raw_WC","VAD"]
noise_opts = ["none","spike","lenschow_linear","lenschow_subrange","lenschow_spectrum"]
vol_opts = ["none","spectral_correction_fit","acf"]
contamination_opts = ["none","taylor_ws","taylor_var"]

for ii in ws_opts:
    for jj in noise_opts:
        for kk in vol_opts:
            for mm in contamination_opts:
                
                #Initialize arrays for the new lidar TI after each correction has been applied 
                TI_WC_noise_all = np.zeros(len(files_WC_DBS))
                TI_WC_noise_all[:] = np.nan
                TI_WC_vol_avg_all = np.zeros(len(files_WC_DBS))
                TI_WC_vol_avg_all[:] = np.nan
                TI_WC_var_contamination_all = np.zeros(len(files_WC_DBS))
                TI_WC_var_contamination_all[:] = np.nan

                p_all = np.zeros(len(files_WC_DBS))
                p_all[:] = np.nan
                
                mode_ws,mode_noise,mode_vol,mode_contamination = [ii,jj,kk,mm]

                for i in range(len(files_WC_DBS)):
                    if ~np.isnan(TI_ref_all[i]):
                        if "raw" in mode_ws:
                            frequency = 1.
                            file_temp = files_WC_DBS[i]
                        else:
                            frequency = 1./4
                            file_temp = files_WC_VAD[i]
    
                        u_rot = np.load(file_temp)['u_rot']
                        u_var = np.load(file_temp)['u_var']
                        U = np.load(file_temp)['U']
                                              
                        p_all[i] = np.load(files_WC_DBS[i])['p']
                            
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
                        

                #Corrected TI is the value of the TI after it has been through all correction modules   
                TI_WC_corrected_all = TI_WC_var_contamination_all
                
                opts = 'WS_' + mode_ws +'_N_' + mode_noise + '_V_' + mode_vol + '_C_' + mode_contamination

                mask = [~np.isnan(TI_ref_all),~np.isnan(TI_WC_corrected_all)]
                mask = reduce(np.logical_and, mask)
                MAE_all = MAE(TI_ref_all[mask],TI_WC_corrected_all[mask])
                        
                        
                #Classify stability by shear parameter, p. A different stability metric could be used if available.                         
                mask = [~np.isnan(TI_ref_all),~np.isnan(TI_WC_corrected_all),p_all >= 0.2]
                mask = reduce(np.logical_and, mask)
                MAE_s = MAE(TI_ref_all[mask],TI_WC_corrected_all[mask])
                
                mask = [~np.isnan(TI_ref_all),~np.isnan(TI_WC_corrected_all),p_all >= 0.1,p_all < 0.2]
                mask = reduce(np.logical_and, mask)
                MAE_n = MAE(TI_ref_all[mask],TI_WC_corrected_all[mask])
                
                mask = [~np.isnan(TI_ref_all),~np.isnan(TI_WC_corrected_all),p_all < 0.1]
                mask = reduce(np.logical_and, mask)
                MAE_u = MAE(TI_ref_all[mask],TI_WC_corrected_all[mask])
                
                if MAE_all < MAE_min:
                    MAE_min = MAE_all
                    opts_best = opts
                if MAE_s < MAE_min_s:
                    MAE_min_s = MAE_s
                    opts_best_s = opts    
                if MAE_n < MAE_min_n:
                    MAE_min_n = MAE_n
                    opts_best_n = opts  
                if MAE_u < MAE_min_u:
                    MAE_min_u = MAE_u
                    opts_best_u = opts
 
                #Write out final MAE values after all corrections have been applied for this model combination
                opts_temp = 'WS_' + mode_ws +'_N_' + mode_noise + '_V_' + mode_vol + '_C_' + mode_contamination
                with open(main_directory + 'L_TERRA_combination_summary_WC.csv', 'a') as fp:
                    a = csv.writer(fp, delimiter=',')
                    data = [[opts_temp,'{:0.2f}'.format(MAE_all),'{:0.2f}'.format(MAE_s),\
                    '{:0.2f}'.format(MAE_n),'{:0.2f}'.format(MAE_u)]]
                    a.writerows(data)   
                               
#Write out minimum MAE values for each stability class and model options associated with these minima                          
with open(main_directory + 'L_TERRA_combination_summary_WC.csv', 'a') as fp:
     a = csv.writer(fp, delimiter=',')
     data = [['Overall Min. MAE','{:0.2f}'.format(MAE_min)]]
     a.writerows(data) 
     data = [['Best Options',opts_best]]
     a.writerows(data) 
     data = [['Overall Min. MAE Stable','{:0.2f}'.format(MAE_min_s)]]
     a.writerows(data) 
     data = [['Best Options Stable',opts_best_s]]
     a.writerows(data) 
     data = [['Overall Min. MAE Neutral','{:0.2f}'.format(MAE_min_n)]]
     a.writerows(data) 
     data = [['Best Options Neutral',opts_best_n]]
     a.writerows(data) 
     data = [['Overall Min. MAE Unstable','{:0.2f}'.format(MAE_min_u)]]
     a.writerows(data) 
     data = [['Best Options Unstable',opts_best_u]]
     a.writerows(data) 


