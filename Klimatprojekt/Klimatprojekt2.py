import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numpy import genfromtxt
import scipy.io as spio

s = 1
CO2_ref = genfromtxt('conc_CO2.csv', delimiter=',')
RFCO2_ref = genfromtxt('RFCO2.csv', delimiter=',')
RFAero_ref = genfromtxt('RFAerosols.csv', delimiter=',')
RFOther_ref = genfromtxt('RFOther.csv', delimiter=',')
B_0 = [600*0.469, 600, 1500]
# Uppgift 8.

def rf_co2(p_co2):
    return 5.35 * np.log(p_co2/B_0[0])

RFCO2_model = rf_co2(CO2_ref)

# Uppgift 9.
RFAeroOther = RFAero_ref * s + RFOther_ref

# Uppgift 10
lam = 0.8 # 0.5-1.3
k = 0.5 # 0.2 - 1
c = 4186 # J/(kg*K)
rho = 1020 # kg/m^3
h = 50 # m
d = 2000 # m
C1 = c * rho * h * 3.16887646 * 10**(-8)  # W*yr/(K*m^2)
C2 = c * rho * d * 3.16887646 * 10**(-8)  # W*yr/(K*m^2)
RF_tot = RFCO2_model + RFAeroOther
t_span2 = np.arange(0, RF_tot.size, 1)
'''
def RF_tot(t):
    return (RFCO2_model + RFAeroOther)[t]
'''

'''
def f(T, t):
    T_1, T_2 = T
    return [RF_tot[t] - T_1/lam - k * (T_1 - T_2), k * (T_1 - T_2)]

for 
    T[n] = T[n-1] + h * f(T[n-1], t_span[n])

def dT_dt(T, t):
    T_1, T_2 = T

    dT_dt = [RF_tot[t] - T_1/lam - k * (T_1 - T_2), k * (T_1 - T_2)]

    return RF_tot - dT_dt

T_0 = [0, 0]
sol = odeint(dT_dt, T_0, t_span2)
'''
time = RFCO2_ref.size+100000
dT1 =[]
dT2 = []
T1 = []
T2 = []
T1.append(0)
T2.append(0)

i = 0
RF = 1
while T2[i] < (1-np.exp(-1)) * RF * lam:
    diff = (T1[i] - T2[i]);

    dT1 = (RF - T1[i]/lam - k * diff)/C1
    dT2 = (k * diff)/C2
    T1.append(T1[i] + dT1)
    T2.append(T2[i] + dT2)
    i = i + 1

e_folding_time = i
print(e_folding_time)




fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(RFCO2_model, label="Model")
ax1.plot(RFCO2_ref, label="Reference")
ax1.plot(RFAeroOther, label="Aerosols + other")

ax1.legend(loc="best")

ax2.plot(T1, label = "Ytvatten")
ax2.plot(T2, label = "Djupvatten")

ax2.legend(loc="best")

plt.show()