import os, numpy as np, agama, scipy.special, scipy.integrate, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pyfalcon
import plotly.express as px

agama.setUnits(length=1, velocity=1, mass=1) #kpc,km/s, Solar mass

# Milky Way potential and distribution function
pot_mw = agama.Potential(
    dict(type='Dehnen', mass=1.0e10, scaleRadius=1.0, gamma=1),  # Bulge
    dict(type='MiyamotoNagai', mass=5.0e10, scaleRadius=3.0, scaleHeight=0.3),  # Disk
    dict(type='NFW', mass=2e12, scaleRadius=22.0))  # Dark matter halo
df_mw = agama.DistributionFunction(type='quasispherical', potential=pot_mw)
r_array  = np.logspace(-1, 2.2, 100)
sigma_array= agama.GalaxyModel(pot_mw,df_mw).moments(np.column_stack((r_array, r_array*0, r_array*0)), dens=False, vel=False, vel2=True)[:,0]**0.5

# Extrapolate sigma(r) using a log-log spline
logspl  = agama.Spline(np.log(r_array), np.log(sigma_array))
sigma   = lambda r: np.exp(logspl(np.log(r)))

# Minicluster parameters
mc_mass0=1e-7
mc_rvir=1e-4
mc_rs=1e-5
Nbody=10000

# Minicluster potential and distribution function
pot_mc  = agama.Potential(type='spheroid', gamma=1.5, beta=3, scaleradius=mc_rs, outercutoffradius=mc_rvir, mass=mc_mass0)
df_mc = agama.DistributionFunction(type='quasispherical', potential=pot_mc)
mc_mass0 = pot_mc.totalMass()

# Sample minicluster particles
mcp_xv, mcp_mass = agama.GalaxyModel(pot_mc, df_mc).sample(Nbody)

# Initial conditions for minicluster center
R0 = 8
Y0 = 0
Z0=-1
F=pot_mw.force(R0,Y0,Z0)
Vx0 = 0
Vy0 = 0
Vz0 = 220
Vx=Vx0
Vy=Vy0
Vz=Vz0
mc_center = np.array([R0, Y0, Z0, Vx, Vy, Vz])
# putting the mc particles in the mw frame
mcp_xv+= mc_center

# Dynamical friction function
def dynfricAccel(pos, vel, mass):
    r   = sum(pos**2)**0.5
    v   = sum(vel**2)**0.5
    rho = pot_mw.density(pos)
    coulombLog = 3.0
    X = v / (2**0.5 * sigma(r))
    return -vel / v * ((4*np.pi * agama.G**2 * mass * rho * coulombLog/ v**2) *
        (scipy.special.erf(X) - 2/np.pi**.5 * X * np.exp(-X**2)) )

# Orbit integration with dynamical friction
def orbitDF(ic, time, timestart, trajsize, mass):
    if mass == 0:
        return agama.orbit(ic=ic, potential=pot_mw, time=time, timestart=timestart, trajsize=trajsize,accuracy=1e-10)
    times = np.linspace(timestart, timestart+time, trajsize)
    sol=solve_ivp(
        lambda t, xv: np.hstack((xv[3:6], 
                                pot_mw.force(xv[0:3], t=t) + 
                                dynfricAccel(xv[0:3], xv[3:6], mass))),
        [timestart, timestart+time],
        ic,
        method='DOP853',
        t_eval=times,
        rtol=1e-14,
        atol=1e-16
    )
    return sol.t, sol.y.T

# Calculate orbital period
orbit_time=pot_mw.Tcirc(mc_center)

# Simulation parameters
simulation_time=2*orbit_time
tupd = 1e-3 # interval for plotting and updating the satellite mass for the restricted N-body simulation
tau  = 1e-4 # timestep of the full N-body sim (typically should be smaller than softening_length/v, where v is characteristic internal velocity)
softening_length  = 1e-5   # softening length for the full N-body simulation

time     = 0.0  
times_t  = [time]
times_u  = [time]
mc_mass   = [mc_mass0]
mc_traj   = [mc_center]
bound_energy  = np.ones(len(mcp_xv), bool)
bound_dist  = np.ones(len(mcp_xv), bool)

time_i=0

# Main simulation loop
while time_i<simulation_time:
    mc_time_center,mc_orbit_center=orbitDF(ic=mc_center,time=tupd,timestart=time_i,trajsize=10000,mass=mc_mass[-1] )
    times_u.append(mc_time_center[-1])
    times_t.extend(mc_time_center[1:])
    mc_traj.extend(mc_orbit_center[1:])
    
    mc_center=mc_orbit_center[-1]
    pot_total=agama.Potential(pot_mw,agama.Potential(potential=pot_mc, center=np.column_stack((mc_time_center, mc_orbit_center))))
    mcp_xv = np.vstack(agama.orbit(ic=mcp_xv, potential=pot_total, time=tupd, timestart=time_i, trajsize=1,accuracy=1e-10)[:,1])
    pot_mc = agama.Potential(type='multipole', particles=(mcp_xv[:,0:3] - mc_center[0:3], mcp_mass), symmetry='a', lmax=4)   
    bound_energy = pot_mc.potential(mcp_xv[:,0:3] - mc_center[0:3]) + 0.5 * np.sum((mcp_xv[:,3:6] - mc_center[3:6])**2, axis=1) < 0
    bound_dist=np.sum((mcp_xv[:,0:3] - mc_center[0:3])**2, axis=1) < mc_rvir**2
    bound=bound_dist& bound_energy
    mc_mass.append(np.sum(mcp_mass[bound])) 
    time_i+=tupd

# Plot orbit and final particle positions
traj = np.vstack(mc_traj)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(traj[:,0], traj[:,2], 'r', linewidth=1, alpha=0.5, label='Orbit')
ax.scatter(mcp_xv[:,0][bound], mcp_xv[:,2][bound], color='green', s=2, label='Bound')
ax.scatter(mcp_xv[:,0][~bound], mcp_xv[:,2][~bound], color='red', s=0.5, alpha=0.5, label='Unbound')
ax.scatter([8], [0], marker='*', s=300, color='gold', edgecolors='orange', label='MW center')

ax.set_xlabel('x [kpc]')
ax.set_ylabel('z [kpc]')
ax.legend()
ax.set_aspect('auto')  
ax.grid(alpha=0.3)

plt.show()

# Print statistics
print(f"\nBound particles: {np.sum(bound)} / {len(mcp_xv)}")
print(f"Bound by distance criterion: {np.sum(bound_dist)} / {len(mcp_xv)}")
print(f"Bound by energy criterion: {np.sum(bound_energy)} / {len(mcp_xv)}")
print(f"Final minicluster mass: {mc_mass[-1]:.4e} / initial {mc_mass0:.4e}")
print(f"Mass loss: {(1 - mc_mass[-1]/mc_mass0)*100:.1f}%")
