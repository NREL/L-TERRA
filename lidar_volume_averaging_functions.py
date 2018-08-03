# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:14:01 2015

@author: jnewman
"""

def spectral_correction(u_rot,frequency,mode_ws,option):
    #Estimates loss of variance due to volume averaging by extrapolating spectrum 
    #out to higher frequencies and integrating spectrum over higher frequencies


    #Inputs
    #u_rot: Time series of streamwise wind speed
    #frequency: Sampling frequency of time series
    #mode_ws: raw_WC, VAD, or raw_ZephIR 
    #option: Type of volume averaging correction to be applied. Options are spectral_correction_fit and acf.

    #Outputs
    #var_diff: Estimate of loss of streamwise variance due to volume averaging  

    import numpy as np
    from lidar_preprocessing_functions import get_10min_spectrum, get_10min_spectrum_WC_raw, get_10min_var
    from lidar_noise_removal_functions import acvf
    from lidar_volume_averaging_functions import Kaimal_spectrum_func, Kaimal_spectrum_func2
    import scipy.signal
    from lmfit import minimize,Parameters
    ten_min_count = frequency*60*10
    
    var_diff = []

    for i in np.arange(0,len(u_rot)-ten_min_count+1,ten_min_count):
        u_temp = u_rot[i:i+ten_min_count]
        U = np.mean(u_temp)
        u_var = get_10min_var(u_temp,frequency)
        #Detrend time series before estimating parameters for modeled spectrum
        u_temp = scipy.signal.detrend(u_temp)
        if "raw_WC" in mode_ws:
            [S,fr] = get_10min_spectrum_WC_raw(u_temp,frequency)
        else:
            [S,fr] = get_10min_spectrum(u_temp,frequency)
        if "spectral_correction_fit" in option:
            #Find value of length scale that produces best fit to idealized
            #Kaimal spectrum
            fit_params = Parameters()
            fit_params.add('L', value=500,min=0,max=1500)
            fit_params.add('U', value=U,vary=False)
            fit_params.add('u_var', value=u_var,vary=False)
            out = minimize(Kaimal_spectrum_func2, fit_params, args=(fr,), kws={'data':fr*S})
            L = out.params['L'].value
            
        else:
            #Otherwise, use ACF to estimate integral length scale and use this 
            #value for the length scale in the Kaimal modeled spectrum
            lags = np.arange(0,ten_min_count)/float(frequency)
            u_corr = acvf(u_temp)
            u_acf = u_corr/u_corr[0]
            indices = np.arange(0,len(u_acf))
            x = indices[np.array(u_acf)<=0]
            #ACF is integrated to the first zero crossing to esimate the integral
            #time scale and multipled by the mean horizontal wind speed to estimate
            #the integral length scale 
            L = np.trapz(u_acf[:x[0]],lags[:x[0]])*U
            
        fr2 = np.linspace(0,float(10)/2,float(6000/2))
        #Calculate Kaimal spectrum from 0 to 5 Hz
        S_model = Kaimal_spectrum_func((fr2,U,u_var),L)
        #Integrate spectrum for frequency values higher than those in the original
        #spectrum from the lidar
        var_diff.append(np.trapz((S_model[fr2 > fr[-1]]/fr2[fr2 > fr[-1]]),fr2[fr2 > fr[-1]]))

            
    return np.array(var_diff) 
       

def Kaimal_spectrum_func(X,L):
    #Given values of frequency (fr), mean horizontal wind speed (U), streamwise 
    #variance (u_var), and length scale (L), calculate idealized Kaimal spectrum
    #This uses the form given by Eq. 2.24 in Burton et al. (2001)
    #Reference: Burton, T., D. Sharpe, N. Jenkins, N., and E. Bossanyi, 2001: 
    #Wind Energy Handbook, John Wiley & Sons, Ltd., 742 pp.
    fr,U,u_var = X
    return u_var*fr*((4*(L/U)/((1+6*(fr*L/U))**(5./3))))
    
    
def Kaimal_spectrum_func2(pars, x, data=None):
    #Kaimal spectrum function for fitting. Trying to minimize the difference 
    #between the actual spectrum (data) and the modeled spectrum (model)
    vals = pars.valuesdict()
    L =  vals['L']
    U =  vals['U']
    u_var = vals['u_var']
    model = u_var*x*((4*(L/U)/((1+6*(x*L/U))**(5./3))))
    if data is None:
        return model
    return model-data   