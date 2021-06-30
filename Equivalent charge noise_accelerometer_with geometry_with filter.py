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
import control

##########  DIGITAL FILTER
# directly copied from VLT-MAN-ESO-15400-4234 : VLTI - UT Vibration Monitoring System, Software User and Maintenance Manual
##########
pi = np.pi
ts = 1/4000;     # sampling time    
w0 = 4.33*2*pi;  # pole
w1 = 5.77*2*pi;  # zero
w3 = 2.5*2*pi;   # highpass
w4 = 1000*2*pi;  # pole
w5 = 150*2*pi;   # zero

g_1 = (w1*ts/2+1)/(w0*ts/2+1)*10**(-0.3/20);
a1_1 = (w1*ts/2-1)/(w1*ts/2+1);
b1_1 = (w0*ts/2-1)/(w0*ts/2+1);
tf_1 = control.tf(g_1*np.array([1, a1_1]),[1, b1_1],ts);

g_2 = ts**2/4/(1+w3*ts/2+(w3*ts/2)**2);
a1_2 = 2;
a2_2 = 1;
b1_2 = 2*((w3*ts/2)**2-1)/(1+w3*ts/2+(w3*ts/2)**2);
b2_2 = (1-w3*ts/2+(w3*ts/2)**2)/(1+w3*ts/2+(w3*ts/2)**2);
tf_2 = control.tf(g_2*np.array([1, a1_2, a2_2]),[1, b1_2, b2_2],ts);

g_3 = 1/(1+w3*ts/2);
a1_3 = -1;
b1_3 = (w3*ts/2-1)/(w3*ts/2+1);
tf_3 = control.tf(g_3*np.array([1, a1_3]),[1, b1_3],ts);

a1_4 = (w5*ts/2-1)/(w5*ts/2+1);
b1_4 = (w4*ts/2-1)/(w4*ts/2+1);
g_4 = (1+b1_4)/(1+a1_4);
tf_4 = control.tf(g_4*np.array([1, a1_4]),[1, b1_4],ts);

tf = tf_1*tf_2*tf_3*tf_4
#inverse transfer function
itf = control.tf(tf.den,tf.num,ts)

##Bode plot  of transfer function 
mag, phase, omega = control.bode(tf, Hz=True, dB=True, omega_limits=(.1, 1000), plot=False)

# %%
#Parameter for noise equivalent circuit
g=9.81 #gravity unit [ms-2]
en=4.9e-6# [V]voltage noise amplifier
Cf=1e-9#[F] feedback capacitance amplifier (it depends on gain--80 dB)
L=40 #[m] Cable length
C_L= 90e-12 #[F/m] cable capacitance per meter
Ca=1.1e-9 #[F] accelerometer intrinsic capacitance
Ct=Ca+C_L*L
Rt=1e9 #[Ohm] usually neglected
sens=10e-12 #sensitivity [C/ms-2]
fmin=2#np.min(omega/(2*pi))#2 #[Hz]
fmax=100e3#np.max(omega/(2*pi))#100e3 #[Hz] from spec
df=np.mean(np.diff(omega/(2*pi))) #1
fr=omega/(2*pi)#np.arange(fmin,fmax, df) #omega/(2*pi)#
Zt=np.abs(Ct+1/(1j*fr*Rt))#(Rt/(1+1j*2*np.pi*fr[i]*Rt*Ct))

#Equivalent charge noise
qt=en*(Cf+Ct)#(1+Cf/(Zt))*Ct #[C]


# Compute noise acceleration
n=np.shape(fr)
acc_noise = qt*np.ones(n)/np.sqrt(fmax-fmin)/sens #/g# np.sqrt(PSD)


#Mirror 3
#Insert mirror geometry
acc_noise_3 = acc_noise*np.sqrt(2)*1e6 #[um/s^2] with geometrical factor 45 degree
#Compute the total mirror acceleration PSD, considering number of accelerometer
psd_acc_3 = (acc_noise_3)**2 / 2 #number of acc= 2 
#Compute the position PSD considering filter mag^2
psd_pos_3=mag**2*psd_acc_3/(2*pi*fr)**4
#Compute reverse cumulated PSD
rcum_psd_pos_3= np.cumsum(psd_pos_3[::-1])[::-1]*df

#Mirror 2
acc_noise_2 = acc_noise*2*1e6 #[um/s^2] with geometrical factor 2 for reflection mirror 
psd_acc_2 = (acc_noise_2)**2/1   #number of accel=1 
psd_pos_2=mag**2*psd_acc_2/(2*pi*fr)**4
rcum_psd_pos_2= np.cumsum(psd_pos_2[::-1])[::-1]*df

#Mirror 1
acc_noise_1 = acc_noise*2*1e6 #[um/s^2] with geometrical factor 
psd_acc_1 = (acc_noise_1)**2 / 4 #number of accel= 4 
psd_pos_1=mag**2*(psd_acc_1/(2*pi*fr)**4)
rcum_psd_pos_1= np.cumsum(psd_pos_1[::-1])[::-1]*df


# Compute single OPD
OPDrms_3=np.sqrt(rcum_psd_pos_3[0])
OPDrms_2=np.sqrt(rcum_psd_pos_2[0])
OPDrms_1=np.sqrt(rcum_psd_pos_1[0])

print("Intrinsic accel noise="+str("%.2f" % (acc_noise[0]*1e6/g))+"µg/$Hz^0.5")
print("OPD_M3= "+str("%.2f" % (OPDrms_3*1000))+" nm")
print("OPD_M2= "+str("%.2f" % (OPDrms_2*1000))+" nm")
print("OPD_M1= "+str("%.2f" % (OPDrms_1*1000))+" nm")

#Calculate total OPD
PSDpos_tot=psd_pos_1+psd_pos_2+psd_pos_3
rcum_psd_pos_tot=np.cumsum(PSDpos_tot[::-1])[::-1]*df
OPD_rms_tot=np.sqrt(rcum_psd_pos_tot[0])
print("OPD_M1= "+str("%.2f" % (OPD_rms_tot*1000))+" nm")



#Plot results
plt.loglog(fr, PSDpos_tot)
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD pos [$µm^2/Hz$]')
plt.xlim(.1,400)
plt.legend(["L="+str(L)+"m"])



# fig.savefig('Noise equivalent charge, 20 m cable.pdf')

plt.show()    

