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
import os
os.chdir('/Users/u0144114/Desktop/Vibration control/pyGravity')
import atmosphere_piston_2

g=9.81 #gravity unit [ms-2]
fig, axarr = plt.subplots(1, sharex=True, sharey='row', figsize=(12, 12), squeeze=False)
uts = [1, 2, 3, 4]
nperseg = 2**13#2**12

######EXPERIMENTAL DATA
#os.chdir("C:/Users/u0144114/Desktop/Vibration control") 
#raw_BNC = np.genfromtxt('noise.txt', delimiter = '', usecols=[0,1])

os.chdir("C:/Users/u0144114/Desktop/Vibration control/Measurements_23_04_21")
raw_ampli= np.genfromtxt('20-20-2.csv', delimiter = ',', usecols=[0,1], skip_header=12)

os.chdir("C:/Users/u0144114/Desktop/Vibration control/Measurements_23_04_21") 
raw_cable = np.genfromtxt('40entire.csv', delimiter = ',', usecols=[0,1], skip_header=12)

os.chdir("C:/Users/u0144114/Desktop/Vibration control/Measurements_12_03_21") 
raw_accel = np.genfromtxt('40m_cable and accel on the table_vibrat on the ground.csv', delimiter = ',', usecols=[0,1], skip_header=12)
#print (raw)

os.chdir("C:/Users/u0144114/Desktop/Vibration control/Manhattan/data") 
for i, ut in enumerate(uts):
    raw_manh = fits.getdata('VIBMAH_FTon_MANon.fits', f'MAH2-UT1')
    
# Drop samples with no timestamps
raw_manh = raw_manh[raw_manh['TIME'] > 0]

# Compute the period
dt = np.mean(np.diff(raw_manh['TIME']))*1e-6

# Compute mirror acceleration
acc = {}
acc[3] = -0.01*(raw_manh['r1'] + raw_manh['r2'])/2.0 #/g
acc[2] = -0.01*raw_manh['r3'] #/g
acc[4] = -0.01*raw_manh['r4'] #/g
acc[1] = +0.01*(raw_manh['r5'] + raw_manh['r6'] + raw_manh['r7'] + raw_manh['r8'])/4.0 #/g

# Convert to piston, based on geometry
acc[1] *= 2.0*1e6
acc[2] *= 2.0*1e6
acc[3] *= np.sqrt(2.0)*1e6
acc[4] *= np.sqrt(2.0)*1e6

psd_acc_e = {}
freq_1, psd_acc_e[1] = sig.welch(acc[1], fs=1/dt, nperseg=nperseg, detrend='linear')
freq_2, psd_acc_e[2] = sig.welch(acc[2], fs=1/dt, nperseg=nperseg, detrend='linear')
freq_3, psd_acc_e[3] = sig.welch(acc[3], fs=1/dt, nperseg=nperseg, detrend='linear')
freq_4, psd_acc_e[4] = sig.welch(acc[4], fs=1/dt, nperseg=nperseg, detrend='linear')

# Double integrate acceleration into position
pos_E1= np.cumsum(np.cumsum(acc[1]))*dt**2
pos_E2 = np.cumsum(np.cumsum(acc[2]))*dt**2
pos_E3 = np.cumsum(np.cumsum(acc[3]))*dt**2
pos_E4 = np.cumsum(np.cumsum(acc[4]))*dt**2


# Compute PSD of acceleration
freq_1, psd_pos_E1 = sig.welch(pos_E1, fs=1/dt, nperseg=nperseg, detrend='linear')
freq_2, psd_pos_E2 = sig.welch(pos_E2, fs=1/dt, nperseg=nperseg, detrend='linear')
freq_3, psd_pos_E3= sig.welch(pos_E3, fs=1/dt, nperseg=nperseg, detrend='linear')
freq_4, psd_pos_E4= sig.welch(pos_E4, fs=1/dt, nperseg=nperseg, detrend='linear')


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


#################### ATMOSPHERE


(fs, atm_pist_psd), atm_params = atmosphere_piston_2.atm_piston()


#%%
################NOISE EQUIVALENT CIRCUIT



Length=[[10,45,10,0,0,0,0,0],[44,45,39,39.5,15,21,21,18],[44,45,39,39.5,0,0,0,0]] #Configurations current,new
number=[[4,1,2,0.1,0.1,0.1,0.1,0.1],[4,1,2,1,1,1,1,1],[4,1,2,1,0.1,0.1,0.1,0.1]]  # Number of accelerometers each mirror
angle=[[0,0,45,45,0,0,0,0],[0,0,45,45,4.38,25.29,7.65,13.26],[0,0,45,45,4.38,25.29,7.65,13.26]]  #Direction of each accelerometer towars the angle
geometry=np.zeros(np.shape(Length))
for k in range(len(Length)):
    for l in range(len(Length[0])):
        geometry[k][l]=2*np.cos(angle[k][l]*pi/180)#[[2,2,np.sqrt(2),np.sqrt(2),2*np.cos(45*),0,0,0],[2,2,np.sqrt(2),np.sqrt(2),2,2,2,2]]

#Parameters for noise equivalent circuit

en=4.9e-6# [V]voltage noise amplifier
Cf=1e-9#[F] feedback capacitance amplifier (it depends on gain--80 dB)
#L_list=[44.,45.,39.] #M1,M2,M3,M4,M5,M6,M7,M8
C_L= 0#50e-12 #[F/m] cable capacitance per meter
Ca=1.1e-9 #[F] accelerometer intrinsic capacitance
Rt=1e9 #[Ohm] usually neglected
sens=10e-12 #sensitivity [C/ms-2]
fmin=1#np.min(omega/(2*pi))#2 #[Hz]
fmax=100e3#np.max(omega/(2*pi))#100e3 #[Hz] from spec
df=np.mean(np.diff(omega/(2*pi))) #1
fr=omega/(2*pi)#np.arange(fmin,fmax, df) #omega/(2*pi)#
n=np.shape(fr)

Ct=np.zeros(np.shape(Length))
Zt=np.zeros(np.shape(Length))
qt=np.zeros(np.shape(Length))  #Equivalent charge noise
acc_noise=np.zeros(np.shape(Length),dtype=object)
psd_acc=np.zeros(np.shape(Length),dtype=object)
psd_pos=np.zeros(np.shape(Length),dtype=object)


for k in range(len(Length)):
    for l in range(len(Length[0])):
        Ct[k][l]=(number[k][l]>=1)*(Ca+C_L*Length[k][l])
#    Ct.append(Ca+C_L*L_list[l])
for k in range(len(Length)):
    for l in range(len(Length[0])):
 #       Zt[k][l]=np.abs(Ct[k][l]+1/(1j*fr*Rt))#(Rt/(1+1j*2*np.pi*fr[i]*Rt*Ct))
        qt[k][l]=en*(Cf+Ct[k][l])#(1+Cf/(Zt))*Ct #[C]

# Compute noise acceleration
for k in range(len(Length)):
    for l in range(len(Length[0])):
        acc_noise[k][l]=(qt[k][l]/np.sqrt(fmax-fmin)/sens)*1e6*geometry[k][l]*np.ones(n) #/g# np.sqrt(PSD) #[um/s^2]

#Compute the mirror acceleration PSD, considering number of accelerometer
for k in range(len(Length)):
    for l in range(len(Length[0])):
        psd_acc[k][l]=(number[k][l]>=1)*((acc_noise[k][l])**2/(number[k][l])) 
        psd_pos[k][l]=mag**2*psd_acc[k][l]#/(2*pi*fr)**4
#    plt.loglog(fr, psd_pos[k][0], label=str(k))
#        rcum_psd_pos[k][l]= np.cumsum(psd_pos_3[::-1])[::-1]*df
for l in range(len(Length[0])):
    plt.loglog(fr, psd_pos[0][l], label=str(k))
#Compute the total telescope psd position and plot for every selected configurations of cable lengths
PSDpos_tot=np.zeros(len(Length),dtype=object)
OPD_rms_tot=np.zeros(len(Length))
for k in range(len(Length)):
    for l in range(len(Length[0])):
        PSDpos_tot[k]+=psd_pos[k][l]
 #   plt.loglog(fr, PSDpos_tot[k], label=str(k))

#Compute total OPD rms    
for k in range(len(Length)):  
    OPD_rms_tot[k]=np.trapz(PSDpos_tot[k],fr)**0.5 
    print("OPD_TOT= "+str("%.2f" % (OPD_rms_tot[k]*1000))+" nm")

#Plot the atmosphere position PSD
plt.loglog(fs, atm_pist_psd,color='k',label='D=8m pupil')
plt.loglog(freq_1, psd_pos_E1, color='tab:blue')
plt.loglog(freq_1, psd_pos_E2, color='tab:orange')
plt.loglog(freq_1, psd_pos_E3, color='tab:green')
plt.loglog(freq_1, psd_pos_E4, color='tab:red')

#LOW NOISE ACCELEROMETER REQUIREMENTS-------------------------------------------------------------------------------------------------------------
acc_low_noise=np.zeros(np.shape(Length[0]),dtype=object)
psd_low_acc=np.zeros(np.shape(Length[0]),dtype=object)
psd_low_pos=np.zeros(np.shape(Length[0]),dtype=object)
for l in range(len(Length[0])):
     acc_low_noise[l]=0.1*geometry[1][l]*np.ones(n) #/g# np.sqrt(PSD) #[um/s^2]
#  #       *np.ones(n)
#Compute the total mirror acceleration PSD, considering number of accelerometer
for l in range(len(Length[0])):
     psd_low_acc[l]=(number[1][l]>=1)*((acc_low_noise[l])**2/(number[1][l])) 
     psd_low_pos[l]=mag**2*psd_low_acc[l]#/(2*pi*fr)**4   
 #    plt.loglog(fr, psd_low_pos[0], label=1, color='r')
     
PSDpos_low_tot=0  
for l in range(len(Length[0])):
    PSDpos_low_tot+=psd_low_pos[l]
    
plt.loglog(fr, PSDpos_low_tot, label=1, color='k')

#---------------------------------------------------------------------------------------------------




# Compute single OPD
# OPDrms=np.zeros()
# OPDrms_3=np.trapz(psd_pos_3,fr)**0.5
# OPDrms_2=np.trapz(psd_pos_2,fr)**0.5
# OPDrms_1=np.trapz(psd_pos_1,fr)**0.5

#%%
# for l in range(len(L_list)):
#     print("Intrinsic accel noise="+str("%.2f" % (acc_noise[l][0]*1e6/g))+"µg/$Hz^0.5$")
# print("OPD_M3= "+str("%.2f" % (OPDrms_3*1000))+" nm")
# print("OPD_M1= "+str("%.2f" % (OPDrms_1*1000))+" nm")

#Calculate total OPD






   
#rcum_psd_pos_tot=np.cumsum(PSDpos_tot[::-1])[::-1]*df




#plt.figure(figsize=(10,7))
#note  we need  a factor of 4pi^2 since we are considering PSD in freq and not angular frequency                                            
# (2*np.pi)**2 *np.sqrt(fs**4 * (np.array(atm_pist_psd) * (atm_params['wvl']/(2*np.pi))**2)),label='atmospheric piston')
""" #to check slopes match theory uncomment this
plt.loglog(fs, np.sqrt(fs**(-17/3)*fs**4),label = r'$f^{2-17/6}$')#label='{}'.format(round(1e6 * 1/9.81 * 10**(-4.5),3))+r'${}\mu g/ \sqrt{Hz}$') 
plt.loglog(fs, np.sqrt(fs**(-8/3)*fs**4),label = r'$f^{2-8/6}$')#label='{}'.format(round(1e6 * 1/9.81 * 10**(-4.5),3))+r'${}\mu g/ \sqrt{Hz}$') 
"""
#plt.axhline(0.3,color='k',linestyle='--',label='B&K accelerometer type 4370')
# plt.ylabel(r'$\mu g\ /\ \sqrt{Hz}}$',fontsize=20)
# plt.xlabel('frequency (Hz)',fontsize=20)
plt.gca().tick_params(labelsize=20)
plt.text(1e-3,1e-2,r'$D={}m, r_0={}m, \tau_0={}ms, L_0={}\ m$'.format(atm_params['diam'],round(atm_params['r_0']*(0.5/2.2),1),round(1e3*atm_params['tau_0']*(0.5/2.2)**(6/5),1), atm_params['L_0'] ),fontsize=20)
#plt.legend(fontsize=20)
plt.grid()
plt.tight_layout()
plt.savefig('accelerometer_requirements_to_reach_atmopshere.png')


#Plot results
#plt.loglog(fr, PSDpos_tot)
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD pos [$µm^2/Hz$]')
plt.xlim(1,400)
plt.ylim(1e-14,1e2)



#PLOT PSD ACCELERATION, compared to experimental data
# plt.loglog(freq_1, psd_acc_e[1], label='M1', color='tab:blue', linewidth=3)
# plt.loglog(freq_2, psd_acc_e[2], label='M2', color='tab:orange', linewidth=3 )
# plt.loglog(freq_3, psd_acc_e[3], label='M3', color='tab:green', linewidth=3)
# plt.loglog(freq_4, psd_acc_e[4], label='M4', color='tab:red', linewidth=3)
# plt.axhline(y=psd_acc[0][0][0]/g**2, color='tab:blue', linewidth=3,linestyle='-')
# plt.axhline(y=psd_acc[0][1][0]/g**2, color='tab:orange', linestyle='-',linewidth=3)
# plt.axhline(y=psd_acc[0][2][0]/g**2, color='tab:blue', linestyle='-',linewidth=3)
# plt.axhline(y=psd_acc[1][0][0]/g**2, color='tab:blue', linewidth=3,linestyle='dotted')
# plt.axhline(y=psd_acc[1][1][0]/g**2, color='tab:orange', linestyle='dotted',linewidth=3)
# plt.axhline(y=psd_acc[1][2][0]/g**2, color='tab:green', linestyle='dotted',linewidth=3)
# plt.legend()
# plt.xlim(1e-1,500)
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Acceleretation PSD [\u03BCg$^2$/Hz]')
# plt.title("UT1")
# fig.savefig('Noise equivalent charge, 20 m cable.pdf')

plt.show()    


