"""
packages used are
numpy=> for fast numerics and statistics
scipy=> Tailored for scientific applications like ODE, interpolation, intergration
matplotlib=> for data ploting
pyunicorn=> non linear statistical methods and time series analyses
basemap=> plotting geoscientific data on map
seaborn => plotting data visualization
"""

"""A stationery Gaussian and Markov Process for simulating noise
The process is the Euler-Maruyama Method for solution of OUprocess
"""
import numpy as np
import matplotlib.pyplot as plt

#Model Parameters
t_0 = 0
t_end = 4
length = 1000
theta = 1.1 #mean reverting parameter
mu = 0.8 #mean
sigma = 0.3 # variance of the process

t = np.linspace(t_0, t_end, length) #time axis
dt = np.mean(np.diff(t))

y = np.zeros(length)
y0 = np.random.normal(loc=0.0, scale=1.0) #initial condition

drift = lambda y,t:theta*(mu-y) #define drift term google to learn lambda
diffusion = lambda y,t: sigma # The diffusion Term
noise = np.random.normal(loc=0.0, scale=1.0, size=length)*np.sqrt(dt) #Weiner Process

# Solve the Stochastic Differential Equation
for i in range(1, length):
    y[i] = y[i-1]+drift(y[i-1],i*dt)*dt + diffusion(y[i-1],i*dt)*noise[i]

plt.plot(t,y)
plt.xlabel("Time")
plt.ylabel("Price Level")
plt.show()
