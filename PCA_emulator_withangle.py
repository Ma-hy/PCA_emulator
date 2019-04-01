
# coding: utf-8

# In[ ]:


import gp_emulator
import matplotlib.pyplot as plt
import numpy as np
import prosail
from numpy import genfromtxt
import time
import os

def gen_samp_noise_angle_blacksoil(nsamples, parameters,p_mins, p_maxs, tts, tto, psi):
    samples, distributions = gp_emulator.create_training_set(parameters, p_mins, p_maxs, n_train=nsamples)
    simu_ref = np.zeros((nsamples, 2101))
    for i in range (nsamples) :
        noise = np.random.normal(0,1,2101)/500.
        simu_ref[i,:] = prosail.run_prosail(samples[i,0], samples[i,1], samples[i,2], samples[i,3], samples[i,4], 
                                            samples[i,5], samples[i,6], samples[i,7], 
                                            0.01, tts=tts, tto=tto, psi=psi,
                                            prospect_version='D',typelidf = 2,
                                            rsoil = 0.,rsoil0=np.zeros(2101))+noise
    for i in range(len(samples)):
        "n", "exp(-cab/100)", "exp(-car/100)", "cbrown", "exp(-50*cw)", "exp(-50*cm)", "exp(-lai/2)", "ala/90.", "bsoil", "psoil"
    samples[:,1] = np.exp(-samples[:,1]/100.)
    samples[:,2] = np.exp(-samples[:,2]/100.)
    samples[:,4] = np.exp(-50.*samples[:,4])
    samples[:,5] = np.exp(-50.*samples[:,5])    
    samples[:,6] = np.exp(-samples[:,6]/2.)
    samples[:,7] = samples[:,7]/90.    
    return simu_ref, samples

def gen_samp_noise_angle_withsoil(nsamples, parameters,p_mins, p_maxs, tts, tto, psi):
    samples, distributions = gp_emulator.create_training_set(parameters, p_mins, p_maxs, n_train=nsamples)
    simu_ref = np.zeros((nsamples, 2101))
    for i in range (nsamples) :
        noise = np.random.normal(0,1,2101)/500.
        simu_ref[i,:] = prosail.run_prosail(samples[i,0], samples[i,1], samples[i,2], samples[i,3], samples[i,4], 
                                            samples[i,5], samples[i,6], samples[i,7], 
                                            0.01, tts=tts, tto=tto, psi=psi,
                                            prospect_version='D',typelidf = 2,
                                            rsoil = samples[i,8], psoil = samples[i,9])+noise
    for i in range(len(samples)):
        "n", "exp(-cab/100)", "exp(-car/100)", "cbrown", "exp(-50*cw)", "exp(-50*cm)", "exp(-lai/2)", "ala/90.", "bsoil", "psoil"
    samples[:,1] = np.exp(-samples[:,1]/100.)
    samples[:,2] = np.exp(-samples[:,2]/100.)
    samples[:,4] = np.exp(-50.*samples[:,4])
    samples[:,5] = np.exp(-50.*samples[:,5])    
    samples[:,6] = np.exp(-samples[:,6]/2.)
    samples[:,7] = samples[:,7]/90.    
    return simu_ref, samples

nsamples1 = 500
nsamples2 = 500
c_v, c_s = 3, 1
nsamples = 1000
parameters = ["n","cab","car","cbrown","cw","cm","lai","ala","bsoil","psoil"]
solar_zens = np.arange(10,61,5)
view_zens  = np.arange(0,16,3)
azimuths   = np.arange(5,181,10)
srf = np.loadtxt("../data/S2A_SRS.csv", skiprows=1, delimiter=",")[100:, :]
srf[:, 1:] = srf[:, 1:]/np.sum(srf[:, 1:], axis=0)
srf_land = srf[:, [ 2, 3, 4, 5, 6, 7, 8, 9, 12, 13]].T

raw1 = genfromtxt("../data/Soil_Spectrum_20170329-0401.csv", delimiter=",")
raw2 = genfromtxt("../data/SoilSpectrum_20170504.csv", delimiter=",")
raw3 = genfromtxt("../data/SoilSpectrum_20170504_WetSoil.csv", delimiter=",")
raw4 = genfromtxt("../data/SoilSpectrum_20170531_DrySoilOnly.csv", delimiter=",")
soil_wv = raw1[:,0]
soil_spec = np.vstack((raw1[soil_wv>=400,1:].T, raw2[soil_wv>=400,1:].T, raw3[soil_wv>=400,1:].T, raw4[soil_wv>=400,1:].T))
tmp = np.sum(soil_spec[:, None, :]*srf_land, axis=2)
soil_ref = tmp[~np.isnan(tmp).any(axis=1)]
U_s, s_s, V_s = np.linalg.svd(soil_ref, full_matrices=False)

for sz in solar_zens:
    for vz in view_zens:
        for az in azimuths:
            
            print('VZ%02d_SZ%02d_AZ%03d'%(vz,sz,az))
            parameters = ["n","cab","car","cbrown","cw","cm","lai","ala","bsoil","psoil"]
            p_mins =     [1.5, 10,    8,     0.2, 0.014,0.003, 0,    25.,  0.5,  0.5]
            p_maxs =     [2.0, 80,    20,    0.4, 0.019,0.005, 8.,   80.,  1.50,   1.]
            srf_land = srf[:, [2, 3, 4, 5, 6, 7, 8, 9, 12, 13]].T
            
            data_name = '../data/PCA_emulator_LAI/PCA_data_%d_VZ%02d_SZ%02d_AZ%03d.npz'%(nsamples1 +nsamples2,vz,sz,az)
            if os.path.isfile(data_name): continue
            veg_spec, veg_samples = gen_samp_noise_angle_blacksoil(10000, parameters,p_mins, p_maxs,tts=sz,tto=vz,psi=az)
            veg_ref = np.sum(veg_spec[:, None, :]*srf_land, axis=2)
            U_v, s_v, V_v = np.linalg.svd(veg_ref, full_matrices=False)
            vec = np.dot(np.diag(np.append(s_v[:c_v],s_s[:c_s])),np.vstack((V_v[:c_v],V_s[:c_s])))
            np.savez(data_name, U_v = U_v, s_v = s_v, V_v = V_v, vec = vec)
            
         
            gp_name = '../data/PCA_emulator_LAI/PCA_gp_%d_VZ%02d_SZ%02d_AZ%03d.npz'%(nsamples1 +nsamples2,vz,sz,az)
            if os.path.isfile(gp_name): continue
            p_mins =     [1.5, 30,    8,     0.2, 0.014,0.004, 0,   45.,  0.5,  0.5]
            p_maxs =     [2.0, 60,    10,    0.4, 0.019,0.005, 7.,   85.,  1.50,   1.]
            simu_ref1, samples1 = gen_samp_noise_angle_withsoil(nsamples1, parameters,p_mins, p_maxs, tts=sz,tto=vz,psi=az) 
            p_mins =     [1.5, 20,    8,     0.2, 0.014,0.004, 5,   45.,  0.5,  0.5]
            p_maxs =     [2.0, 60,    10,    0.4, 0.019,0.005, 7.,   85.,  1.50,   1.]
            simu_ref2, samples2 = gen_samp_noise_angle_withsoil(nsamples2, parameters,p_mins, p_maxs, tts=sz,tto=vz,psi=az) 
            simu_ref, samples = np.vstack([simu_ref1,simu_ref2]),np.vstack([samples1,samples2])
            srf_land = srf[:, [2, 3, 4, 5, 6, 7, 8, 9, 12, 13]].T
            integrated_spectra = np.sum(simu_ref[:, None, :]*srf_land, axis=2)  
            
            result = np.dot(np.dot(integrated_spectra,vec.T),np.linalg.inv(np.dot(vec,vec.T)))
            in_pca_emu = gp_emulator.GaussianProcess(inputs=result[:nsamples],targets= samples[:nsamples,6])
            in_pca_emu.learn_hyperparameters(n_tries=15, verbose=True)
            
            np.savez('../data/Bidirec_sample/bidirec_samp_%d_VZ%02d_SZ%02d_AZ%03d.npz'%(nsamples1 +nsamples2,vz,sz,az),
                    veg_ref =veg_ref, veg_samples=veg_samples,simu_spec=integrated_spectra, simu_samples=samples)
            in_pca_emu.save_emulator(gp_name)



