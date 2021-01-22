"""
Speeding up ode integration using numba and Ray
"""

import numpy as np
from scipy.integrate import solve_ivp
import time

import ray
from numba import jit


# start ray
ray.init()


# ========================================================================== #
# function for integration
def twobody(t, state, gm):
	x,y,z = state[0], state[1], state[2]
	vx,vy,vz = state[3], state[4], state[5]
	r = np.sqrt(x**2 + y**2 + z**2)
	ax = -(gm/r**3)*x
	ay = -(gm/r**3)*y
	az = -(gm/r**3)*z
	return np.array([ vx, vy, vz, ax, ay, az ])


@jit(nopython=True)
def twobody_jit(t, state, gm):
	x,y,z = state[0], state[1], state[2]
	vx,vy,vz = state[3], state[4], state[5]
	r = np.sqrt(x**2 + y**2 + z**2)
	ax = -(gm/r**3)*x
	ay = -(gm/r**3)*y
	az = -(gm/r**3)*z
	return np.array([ vx, vy, vz, ax, ay, az ])


# ========================================================================== #
# define initial conditions
gm = 398600.44
state0 = np.array([7000.0, 200.0, -4000.0, 0.0, 7.8, 0.0])
sigma_r = 3.0  # [km]
sigma_v = 0.1  # [km/sec]
period = 3600.0
ics = []
N_ics = 10000
for idx in range(N_ics):
	statep = state0 + np.concatenate((np.random.normal(size=(3))*sigma_r, np.random.normal(size=(3))*sigma_v), axis=0)
	ics.append( statep )


# define function used with single process
def compute_trajectory(ics, tf, gm):
	results_single = []
	for ic in ics:
		sol = solve_ivp(twobody_jit, (0,tf), ic, args=(gm,), method="LSODA")
		statef = np.array([sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1], sol.y[4][-1], sol.y[5][-1]])
		results_single.append( statef )
	return results_single


# define function to be parallelised
@ray.remote
def compute_trajectory_parallel(ics, tf, gm):
	results_mp = []
	for ic in ics:
		sol = solve_ivp(twobody_jit, (0,tf), ic, args=(gm,), method="LSODA")
		statef = np.array([sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1], sol.y[4][-1], sol.y[5][-1]])
		results_mp.append( statef )
	return results_mp
	

# ========================================================================== #
# single process
print(f"Starting single process...")
tstart_single = time.time()

results_single = compute_trajectory(ics, period, gm)

tend_single = time.time()
dt_single = tend_single - tstart_single
print(f"Single process: {dt_single:2.4f} sec")


# ========================================================================== #
# multiple process
print(f"Starting multiple process...")
tstart_mp = time.time()

results_parallel = compute_trajectory_parallel.remote(ics, period, gm)

tend_mp = time.time()
dt_mp = tend_mp - tstart_mp
print(f"Multiple process: {dt_mp:2.4f} sec")

