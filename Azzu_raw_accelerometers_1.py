# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 10:10:49 2021

@author: Azzurra
"""

from astropy.io import fits
import numpy as np
from scipy import signal as sig
from matplotlib import pyplot as plt
import os,subprocess,glob


fig, axarr = plt.subplots(3, sharex=True, sharey='row', figsize=(18, 12), squeeze=False)

#filename = './data/just_a_test.fits'
uts = [1, 2, 3, 4]
nperseg = 2**20 #2**12

# %%
os.chdir("/Users/Azzurra/Desktop/Vibration control/Measurements_23_04_21") 
raw_BNC = np.genfromtxt('20-20connector.csv', delimiter = ',', usecols=[0,1], skip_header=12)

os.chdir("/Users/Azzurra/Desktop/Vibration control/Measurements_23_04_21")
raw_ampli= np.genfromtxt('20-20-2.csv', delimiter = ',', usecols=[0,1], skip_header=12)

os.chdir("/Users/Azzurra/Desktop/Vibration control/Measurements_23_04_21") 
raw_cable = np.genfromtxt('40entire.csv', delimiter = ',', usecols=[0,1], skip_header=12)

os.chdir("/Users/Azzurra/Desktop/Vibration control/Measurements_12_03_21") 
raw_accel = np.genfromtxt('40m_cable and accel on the table_vibrat on the ground.csv', delimiter = ',', usecols=[0,1], skip_header=12)
#print (raw)

os.chdir("/Users/Azzurra/Desktop/Vibration control/Manhattan/data") 
for i, ut in enumerate(uts):
    raw_manh = fits.getdata('VIBMAH_FTon_MANon.fits', f'MAH2-UT1')


# %%

# Drop samples with no timestamps
raw_BNC = raw_BNC[raw_BNC[:,0] > 0]
raw_ampli = raw_ampli[raw_ampli[:,0] > 0]
raw_cable = raw_cable[raw_cable[:,0] > 0]
raw_accel = raw_accel[raw_accel[:,0] > 0]
raw_manh = raw_manh[raw_manh['TIME'] > 0]

# Compute the period
dt_BNC = np.mean(np.diff(raw_BNC[:,0]))
dt_ampli = np.mean(np.diff(raw_ampli[:,0]))
dt_cable = np.mean(np.diff(raw_cable[:,0]))
dt_accel = np.mean(np.diff(raw_accel[:,0]))
dt_manh = np.mean(np.diff(raw_manh['TIME']))*1e-6
#print(dt_BNC, dt_ampli, dt_cable, dt_accel)

# Compute mirror acceleration
acc_BNC = (raw_BNC[:,1])*0.01
acc_ampli = (raw_ampli[:,1])*0.01 #gain 100 V/ms-2
acc_cable = (raw_cable[:,1])*0.01
acc_accel = (raw_accel[:,1])*0.01
acc_manh = -0.01*raw_manh['r3']

# Convert to piston, based on geometry
acc_BNC *= 2*1e6
acc_ampli *= 2*1e6  # [um/s^2]
acc_cable *= 2*1e6  # [um/s^2]
acc_accel *= 2*1e6
acc_manh *= np.sqrt(2.0)*1e6

# Double integrate acceleration into position
pos_BNC= np.cumsum(np.cumsum(acc_BNC))*dt_BNC**2
pos_ampli = np.cumsum(np.cumsum(acc_ampli))*dt_ampli**2
pos_cable = np.cumsum(np.cumsum(acc_cable))*dt_cable**2
pos_accel = np.cumsum(np.cumsum(acc_accel))*dt_accel**2
pos_manh = np.cumsum(np.cumsum(acc_manh))*dt_manh**2

# Compute PSD of acceleration
freq_BNC, psd_acc_BNC = sig.welch(acc_BNC, fs=1/dt_BNC, nperseg=nperseg, detrend='linear')
freq_BNC, psd_pos_BNC = sig.welch(pos_BNC, fs=1/dt_BNC, nperseg=nperseg, detrend='linear')
freq_ampli, psd_acc_ampli = sig.welch(acc_ampli, fs=1/dt_ampli, nperseg=nperseg, detrend='linear')
freq_ampli, psd_pos_ampli = sig.welch(pos_ampli, fs=1/dt_ampli, nperseg=nperseg, detrend='linear')
freq_cable, psd_acc_cable = sig.welch(acc_cable, fs=1/dt_cable, nperseg=nperseg, detrend='linear')
freq_cable, psd_pos_cable= sig.welch(pos_cable, fs=1/dt_cable, nperseg=nperseg, detrend='linear')
freq_accel, psd_acc_accel = sig.welch(acc_accel, fs=1/dt_accel, nperseg=nperseg, detrend='linear')
freq_accel, psd_pos_accel= sig.welch(pos_accel, fs=1/dt_accel, nperseg=nperseg, detrend='linear')
freq_manh, psd_acc_manh = sig.welch(acc_manh, fs=1/dt_manh, nperseg=nperseg, detrend='linear')
freq_manh, psd_pos_manh= sig.welch(pos_manh, fs=1/dt_manh, nperseg=nperseg, detrend='linear')

# Compute frequency period
df_BNC = np.mean(np.diff(freq_BNC))
df_ampli = np.mean(np.diff(freq_ampli))
df_cable = np.mean(np.diff(freq_cable))
df_accel = np.mean(np.diff(freq_accel))
df_manh = np.mean(np.diff(freq_manh))
# Compute reverse cumulated PSD of position
rcum_psd_pos_BNC = np.cumsum(psd_pos_BNC[::-1])[::-1]*df_BNC
rcum_psd_pos_ampli = np.cumsum(psd_pos_ampli[::-1])[::-1]*df_ampli
rcum_psd_pos_cable = np.cumsum(psd_pos_cable[::-1])[::-1]*df_cable
rcum_psd_pos_accel = np.cumsum(psd_pos_accel[::-1])[::-1]*df_accel
rcum_psd_pos_manh = np.cumsum(psd_pos_manh[::-1])[::-1]*df_manh

# Plot results
axarr[0,0].set_title('Vibrations close to cable_Gain 100 V/ms-2')
axarr[0,0].loglog(freq_BNC, psd_acc_BNC, label='20m+20m', color='black', linewidth=.8)
axarr[0,0].loglog(freq_ampli, psd_acc_ampli, label='20m+20m_2', color='red', linewidth=.8)
axarr[0,0].loglog(freq_cable, psd_acc_cable, label='40m', color='blue', linewidth=.8 )
#axarr[0,0].loglog(freq_manh, psd_acc_manh, label='UT1-M3', color='red', linewidth=.8)
axarr[1,0].loglog(freq_BNC, psd_pos_BNC,  color='black', linewidth=.8 )
axarr[1,0].loglog(freq_ampli, psd_pos_ampli, color='red', linewidth=.8)
axarr[1,0].loglog(freq_cable, psd_pos_cable, color='blue', linewidth=.8  )
#axarr[1,0].loglog(freq_manh, psd_pos_manh,color='red', linewidth=.8 )
axarr[2,0].loglog(freq_BNC, rcum_psd_pos_BNC**2, color='black', linewidth=.8)
axarr[2,0].loglog(freq_ampli, rcum_psd_pos_ampli**2, color='red', linewidth=.8)
axarr[2,0].loglog(freq_cable, rcum_psd_pos_cable**2, color='blue', linewidth=.8 )
#axarr[2,0].loglog(freq_manh, rcum_psd_pos_manh**2,color='red', linewidth=.8)

axarr[0,0].legend(loc='upper left')
axarr[1,0].legend(loc='upper right')
axarr[2,0].legend(loc='upper right')
axarr[2,0].set_xlabel('Frequency [Hz]')
    
axarr[0, 0].set_ylabel('PSD Acceler [µm^2/s^3]')
axarr[1, 0].set_ylabel('PSD Pos [µm^2/Hz]')
axarr[2, 0].set_ylabel('rv PSD Pos [µm^2]')

axarr[0, 0].set_xlim(5, 400)
axarr[1, 0].set_xlim(5, 400)
axarr[2, 0].set_xlim(5, 400)

axarr[0, 0].set_ylim(1e-1, 1e10)
axarr[1, 0].set_ylim(1e-15, 1e1)
axarr[2, 0].set_ylim(1e-25, 1e0)

#fig.savefig('Fig-RawAccelerometers.pdf')

plt.show()    

