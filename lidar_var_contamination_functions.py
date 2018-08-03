# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:38:28 2017

@author: jnewman
"""

def var_correction(vr_n,vr_e,vr_s,vr_w,vr_z,wd,U,height_needed,frequency_vert_beam,el_angle,mode):
    #Uses Taylor's frozen turbulence hypothesis with data from the vertically
    #pointing beam to estimate new values of the u and v variance.

    #Inputs
    #vr_n, vr_e, vr_s, vr_w, vr_z: Time series of radial velocity from north-, east-, south, west-, and 
    #vertically pointing beams, respectively, at height of interest.
    #wd: 10-min. Mean wind direction
    #U: 10-min. Mean horizontal wind speed
    #height_needed: Measurement height corresponding to velocity data
    #frequency_vert_beam: Sampling frequency of data from vertically pointing beam
    #el_angle: Elevation angle of off-vertical beam positions (in degrees, measured from the ground)
    #mode: Type of variance contamination correction to be applied. Options are taylor_ws and taylor_var. 

    #Outputs
    #var_diff: Estimate of increase in streamwise variance due to variance contamination

    import numpy as np    
    from lidar_preprocessing_functions import get_10min_var, get_10min_covar
    w_N = np.zeros(len(vr_z))
    w_N[:] = np.nan
    w_E = np.zeros(len(vr_z))
    w_E[:] = np.nan
    w_S = np.zeros(len(vr_z))
    w_S[:] = np.nan
    w_W = np.zeros(len(vr_z))
    w_W[:] = np.nan
    
    u_bar = np.sin(np.radians(wd - 180))*U
    v_bar = np.cos(np.radians(wd - 180))*U
    
    delta_t_vert_beam = 1./frequency_vert_beam
    
    #Calculate the number of time steps needed for eddies to travel from one
    #side of the scanning circle to the other 
    dist = height_needed/np.tan(np.radians(el_angle))
    delta_t_u = dist/u_bar
    interval_u = round(delta_t_u/delta_t_vert_beam)
    delta_t_v = dist/v_bar
    interval_v = round(delta_t_v/delta_t_vert_beam)
    
    #Estimate values of w at different sides of the scanning circle by using 
    #Taylor's frozen turbulence hypothesis 
    for i in range(len(vr_z)):
        try:
            w_N[i] = vr_z[i-interval_v]
            w_E[i] = vr_z[i-interval_u]
        except:
            w_N[i] = np.nan
            w_E[i] = np.nan
        try:
            w_S[i] = vr_z[i+interval_v]
            w_W[i] = vr_z[i+interval_u]
        except:
            w_S[i] = np.nan
            w_W[i] = np.nan
    
    if "taylor_ws" in mode:
        #Use the new values of w to estimate the u and v components using the DBS technique
        #and calculate the variance 
        u_DBS_new = ((vr_e-vr_w) - (w_E-w_W)*np.sin(np.radians(el_angle)))/(2*np.cos(np.radians(el_angle)))
        v_DBS_new = ((vr_n-vr_s) - (w_N-w_S)*np.sin(np.radians(el_angle)))/(2*np.cos(np.radians(el_angle)))   
        
        u_var_lidar_new = get_10min_var(u_DBS_new,frequency_vert_beam)
        v_var_lidar_new = get_10min_var(v_DBS_new,frequency_vert_beam)
    else:
        #Calculate change in w across the scanning circle in north-south and east-west directions
        dw_est1 = w_S - w_N
        dw_est2 = w_W - w_E
        
        vr1_var = get_10min_var(vr_n,1./4)
        vr2_var = get_10min_var(vr_e,1./4)
        vr3_var = get_10min_var(vr_s,1./4)
        vr4_var = get_10min_var(vr_w,1./4)
        dw_var1 = get_10min_var(dw_est1,1./4)
        dw_var2 = get_10min_var(dw_est2,1./4)
        
        vr1_vr3_var = get_10min_covar(vr_n,vr_s,1./4)
        vr2_vr4_var = get_10min_covar(vr_e,vr_w,1./4)
        
        vr1_dw_var = get_10min_covar(vr_n,dw_est1,1./4)
        vr3_dw_var = get_10min_covar(vr_s,dw_est1,1./4)
        vr2_dw_var = get_10min_covar(vr_e,dw_est2,1./4)
        vr4_dw_var = get_10min_covar(vr_w,dw_est2,1./4)
        
        #These equations are adapted from Newman et al. (2016), neglecting terms involving
        #du or dv, as these terms are expected to be small compared to dw
        #Reference: Newman, J. F., P. M. Klein, S. Wharton, A. Sathe, T. A. Bonin,
        #P. B. Chilson, and A. Muschinski, 2016: Evaluation of three lidar scanning 
        #strategies for turbulence measurements, Atmos. Meas. Tech., 9, 1993-2013.
        u_var_lidar_new = (1./(4*np.cos(np.radians(el_angle))**2))*(vr2_var + vr4_var- 2*vr2_vr4_var + 2*vr2_dw_var*np.sin(np.radians(el_angle)) \
        - 2*vr4_dw_var*np.sin(np.radians(el_angle)) + dw_var2*np.sin(np.radians(el_angle))**2)
        
        v_var_lidar_new = (1./(4*np.cos(np.radians(el_angle))**2))*(vr1_var + vr3_var- 2*vr1_vr3_var + 2*vr1_dw_var*np.sin(np.radians(el_angle)) \
        - 2*vr3_dw_var*np.sin(np.radians(el_angle)) + dw_var1*np.sin(np.radians(el_angle))**2)
    
    #Rotate the variance into the mean wind direction
    #Note: The rotation should include a term with the uv covariance, but the 
    #covariance terms are also affected by variance contamination. In Newman 
    #et al. (2016), it was found that the uv covariance is usually close to 0 and
    #can safely be neglected. 
    #Reference: Newman, J. F., P. M. Klein, S. Wharton, A. Sathe, T. A. Bonin,
    #P. B. Chilson, and A. Muschinski, 2016: Evaluation of three lidar scanning 
    #strategies for turbulence measurements, Atmos. Meas. Tech., 9, 1993-2013.
    u_rot_var_new = u_var_lidar_new*(np.sin(np.radians(wd)))**2 + v_var_lidar_new*(np.cos(np.radians(wd)))**2
    
    #Calculate the wind speed and variance if w is assumed to be the same on all 
    #sides of the scanning circle 
    u_DBS = (vr_e-vr_w)/(2*np.cos(np.radians(el_angle)))
    v_DBS = (vr_n-vr_s)/(2*np.cos(np.radians(el_angle))) 
    
    u_var_DBS = get_10min_var(u_DBS,frequency_vert_beam)
    v_var_DBS = get_10min_var(v_DBS,frequency_vert_beam)
    
    u_rot_var = u_var_DBS*(np.sin(np.radians(wd)))**2 + v_var_DBS*(np.cos(np.radians(wd)))**2
    
    return u_rot_var-u_rot_var_new

   
    

