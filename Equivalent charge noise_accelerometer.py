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
nperseg = 2**12#2**12

# %%
os.chdir("/Users/Azzurra/Desktop/Vibration control") 
raw_BNC = np.genfromtxt('noise.txt', delimiter = '', usecols=[0,1])

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
#Parameter for noise equivalent circuit
en=4.9e-6# [V]voltage noise amplifier
Cf=1e-9#[F] feedback capacitance amplifier (it depends on gain--80 dB)
L=20 #[m] Cable length
C_L= 90e-12 #[F/m] cable capacitance per meter
Ca=1.1e-9 #[F] accelerometer intrinsic capacitance
Ct=Ca+C_L*L
Rt=1e9 #[Ohm] usually neglected
sens=10e-12 #sensitivity [C/ms-2]
fmin=2 #[Hz]
fmax=100e3 #[Hz] from spec
fr=np.arange(fmin,fmax, 1)
Zt=np.abs(Ct+1/(1j*fr*Rt))#(Rt/(1+1j*2*np.pi*fr[i]*Rt*Ct))
qt=en*(1+Cf/(Zt))*Ct


# Compute noise acceleration
n=np.shape(fr)
acc_noise = qt*np.ones(n)/np.sqrt(fmax)/sens#  np.sqrt(PSD)
acc_noise *= 1e6 #[um/s^2]

# Compute PSD of acceleration
psd_acc = (acc_noise)**2
psd_pos_acc=psd_acc/(2*np.pi*fr)**4


# Plot results
axarr[0,0].set_title('Equivalent noise charge, gain 80 dB')
axarr[0,0].loglog(fr, acc_noise, label='20m cable length', color='black', linewidth=1)
#axarr[0,0].loglog(freq_ampli, psd_acc_ampli, label='20m+20m_2', color='red', linewidth=.8)
#axarr[0,0].loglog(freq_cable, psd_acc_cable, label='40m', color='blue', linewidth=.8 )
#axarr[0,0].loglog(freq_manh, psd_acc_manh, label='UT1-M3', color='red', linewidth=.8)
axarr[1,0].loglog(fr, psd_acc,  color='black', linewidth=1 )
#axarr[1,0].loglog(freq_ampli, psd_pos_ampli, color='red', linewidth=.8)
#axarr[1,0].loglog(freq_cable, psd_pos_cable, color='blue', linewidth=.8  )
#axarr[1,0].loglog(freq_manh, psd_pos_manh,color='red', linewidth=.8 )
axarr[2,0].loglog(fr, psd_pos_acc, color='black', linewidth=1)
#axarr[2,0].loglog(freq_ampli, rcum_psd_pos_ampli**2, color='red', linewidth=.8)
#axarr[2,0].loglog(freq_cable, rcum_psd_pos_cable**2, color='blue', linewidth=.8 )
#axarr[2,0].loglog(freq_manh, rcum_psd_pos_manh**2,color='red', linewidth=.8)

axarr[0,0].legend(loc='upper left')
axarr[1,0].legend(loc='upper right')
axarr[2,0].legend(loc='upper right')
axarr[2,0].set_xlabel('Frequency [Hz]')
    
axarr[0, 0].set_ylabel('Acceler [µm^2 s^-2/Hz^0.5]')
axarr[1, 0].set_ylabel('PSD Accel [µm^2 s^-2/Hz]')
axarr[2, 0].set_ylabel('PSD Pos [µm^2/Hz]')

axarr[0, 0].set_xlim(fmin, fmax)
axarr[1, 0].set_xlim(fmin, fmax)
axarr[2, 0].set_xlim(fmin, fmax)

axarr[0, 0].set_ylim(1e-1, 2e1)
axarr[1, 0].set_ylim(1e0, 1e2)
axarr[2, 0].set_ylim(1e-20, 1e-2)

fig.savefig('Noise equivalent charge, 20 m cable.pdf')

plt.show()    

