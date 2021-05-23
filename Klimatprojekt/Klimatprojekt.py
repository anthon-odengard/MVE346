import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numpy import genfromtxt
import scipy.io as spio
#U = pd.read_csv('utslapp_CO2.csv')
U = genfromtxt('utslapp_CO2.csv', delimiter=',')
CO2_ref = genfromtxt('conc_CO2.csv', delimiter=',')
beta = 0.41
B_0 = [600*0.469, 600, 1500]
F = [[0, 60, 0], [15, 0, 45], [45, 0, 0]]
NPP0 = 60
F = np.array(F)
t_span = np.arange(0, U.size, 1)

alpha = np.zeros((3,3))
for i in range(0,3):
    for j in range(0,3):
        alpha[i,j] = F[i,j] / B_0[i]

# Uppgift 1
def NPP(B):
    return NPP0 * (1 + beta*np.log(B/B_0[0]))

def dB_dt(B, t):
    B0, B1, B2 = B

    dB1_dt = (alpha[2, 0]*B2 + alpha[1, 0]*B1 - NPP(B0) + U[int(t)]) * 0.469
    dB2_dt = NPP(B0) - alpha[1, 2]*B1 - alpha[1, 0]*B1
    dB3_dt = alpha[1, 2]*B1 - alpha[2, 0]*B2

    return [dB1_dt , dB2_dt, dB3_dt]
    
sol = odeint(dB_dt, B_0, t_span)

## Resultat ovan skiljer sig från referens eftersom det finns approximationer i
## beta och NPP-definitionen, dessutom finns ej interaktion med havet än

# Uppgift 3
A = [0.113, 0.213, 0.258, 0.273, 0.1430]

tau_0 = [2.0, 12.2, 50.4, 243.3, float('inf')]
k = 0.025
#k = 1
def tau(t, emission):
    tau = np.zeros(len(tau_0))
    for i in range(0,len(tau_0)):
        tau[i] = tau_0[i] * (1 + k * np.sum(emission[0:t-1]))
    return tau

# Om det är olika t så är det värt att skriva i labb PM....
def impulse_response(t1, t2, emission):
    prod = A*np.exp(-t1/tau(t2, emission))
    impulse = sum(prod)
    return impulse

time = 500
impulse = np.zeros(time)
for t in range(0, time):
    impulse[t] = impulse_response(t, U.size, U)

# Uppgift 4
M_0 = B_0[0]
def co2_conc_faltning(t, emission):
    sum = 0
    for t_tilde in range(0,t):
        sum = sum + impulse_response(t-t_tilde, t, emission) * emission[t_tilde]
    return M_0 + sum * 0.469
time2 = U.size


sol_sea = np.zeros(time2)
for t in range(0, time2):
    sol_sea[t] = co2_conc_faltning(t, U)
# Uppgift 6
def new_U(t_final):
    U2 =np.zeros(t_final)
    for i in range(0, t_final):
        B0, B1, B2 = sol[i,:]
        U2[i] = (alpha[2, 0]*B2 + alpha[1, 0]*B1 - NPP(B0) + U[int(i)])
    return U2


sol_total = np.zeros(time2)
U2 = new_U(time2)
for t in range(0, time2):
    sol_total[t] = co2_conc_faltning(t, U2)



fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)

ax1.plot(t_span, sol[:, 0], label='Model_atm')
ax1.plot(t_span, CO2_ref, label='Reference_atm')
ax1.plot(t_span, sol[:, 1], label='Model_bio')
ax1.plot(t_span, sol[:, 2], label='Model_ground')
ax1.legend(loc='best')

ax2.plot(impulse, label='impulse')
ax2.legend(loc='best')

ax3.plot(t_span, sol_sea, label='Sea_model_CO2')
ax3.plot(t_span, CO2_ref, label='Reference')
ax3.legend(loc='best')

ax4.plot(t_span, sol_total, label='Total_CO2')
ax4.plot(t_span, CO2_ref, label='Reference')
ax4.legend(loc='best')

index_2100 = U.size - 400
co2_atm_2100 = sol_total[index_2100]
X = np.arange(3)
ax5.bar(X-0.25, B_0, width=0.35)
ax5.bar(X+0.25, [co2_atm_2100, sol[index_2100,1], sol[index_2100, 2]], width=0.35)



plt.show()


