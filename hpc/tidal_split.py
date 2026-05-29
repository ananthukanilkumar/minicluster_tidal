import numpy as np
import agama
import h5py
import matplotlib.pyplot as plt
import sys
sys.path.append('/suphys/aani0116/Work/minicluster_tidal/Tidal_stream')
sys.path.append('/suphys/aani0116/Work/minicluster_tidal/stellar')
print('Test:pi=',np.pi)
from reverse_orbit import reverse_orbit

agama.setUnits(length=1, velocity=1, mass=1)  # kpc, km/s, Solar mass



pot_mw = agama.Potential('/suphys/aani0116/minicluster/minicluster_tidal/Agama/data/McMillan17.ini')

class Minicluster:
    def __init__(self, mass, delta, concentration):
        self.mass          = mass
        self.delta         = delta
        self.concentration = concentration

        self.density = self._density()
        self.radius  = self._radius()
        self.radius_char = self.radius_char()
        self.density_char = self.density_char()


    def _density(self):
        return 2.4e6 * self.delta**3 * (1 + self.delta) * 3e7  #1Gev/cm3=3.10^7 Msun/kpc3

    def _radius(self):
        return (3 * self.mass / (4 * np.pi * self.density)) ** (1/3)

    def density_char(self):
        c = self.concentration
        return self.density * (c**3/3) / (np.log(1 + c) - c / (1 + c))
    
    def radius_char(self):
        c = self.concentration
        return self.radius / c
    
    def mass_within_radius(self, r):
        c = self.concentration
        r_char = self.radius_char 
        r_vir=self.radius
        cx=r/r_char
        return 4*np.pi*self.density_char*(r_vir/c)**3 * (np.log(1 + cx) - cx / (1 + cx)) 

    

mc = Minicluster(mass=1e-6, delta=10, concentration=10)
Nbody=1000

pot_mc = agama.Potential(
    type='Multipole',
    density='Spheroid',
    gamma=1, beta=3,alpha=1,cutoffStrength=1,
    scaleRadius=mc.radius_char,
    outerCutoffRadius=mc.radius,
    mass=mc.mass,
    rmin=0,
    rmax=0,
    gridSizeR=100,
    lmax=0,
)

df_mc   = agama.DistributionFunction(type='quasispherical', potential=pot_mc)
mcp_xv, mcp_mass = agama.GalaxyModel(pot_mc, df_mc).sample(Nbody)
mxp_xv_initial = mcp_xv.copy()

solar_pos = np.array([8.0, 0.0, 0.0, 5.0, 5.0, 220.0]) 
N_orbits=1
orbit_time = N_orbits *    pot_mw.Tcirc(solar_pos)

mc_center=reverse_orbit(solar_pos, pot_mw,N_orbits*pot_mw.Tcirc(solar_pos),4, 10)


mcp_xv += mc_center 

#simulation parameters
simulation_time = orbit_time
tupd = 1e-8 # Gyr
time_i = 0


#stuff for saving data
n_snapshots = 10
snapshot_interval = int(simulation_time / tupd / n_snapshots)
step = 0
mcp_xv_snapshots=[]
bound_snapshots, E_p_snapshots, mc_xv_snapshots, snap_times = [], [], [], []



mc_t=[]
mc_xv=[]
E_p=[]

while time_i < simulation_time:

    mc_time_center, mc_orbit_center = agama.orbit(ic=mc_center, potential=pot_mw, time=tupd,timestart=time_i, trajsize=10, accuracy=1e-10)
    mc_t.extend([mc_time_center[-1]])
    mc_xv.append(mc_orbit_center[-1])


    pot_total = agama.Potential(pot_mw,agama.Potential(potential=pot_mc,center=np.column_stack((mc_time_center, mc_orbit_center))))

    mcp_xv = np.vstack(agama.orbit(ic=mcp_xv, potential=pot_total,time=tupd, timestart=time_i, trajsize=1, accuracy=1e-10,dtype=float)[:, 1])

    
    rel_pos = mcp_xv[:, :3] - mc_orbit_center[-1][:3]
    rel_vel = mcp_xv[:, 3:6] - mc_orbit_center[-1][3:6]
    rel_xv=mcp_xv-mc_orbit_center[-1]

    E_var=0.5*np.linalg.norm(rel_vel, axis=1)**2+pot_mc.potential(rel_pos)
    E_p.append(E_var)
    bound=E_var<0
    n_bound=np.sum(bound)
    print(n_bound)
    if n_bound>10:
        pot_mc=agama.Potential(type='Plummer',mass=sum(mcp_mass[bound]),scaleRadius=mc.radius_char)
    if n_bound<=10:
        pot_mc=agama.Potential(type='Plummer', mass=0.0, scaleRadius=1.0)

    

    if step % snapshot_interval == 0:
        mcp_xv_snapshots.append(rel_xv.copy())
        bound_snapshots.append(bound.copy())
        E_p_snapshots.append(E_var.copy())
        mc_xv_snapshots.append(mc_orbit_center[-1].copy())
        snap_times.append(time_i)
        

    
    mc_center = mc_orbit_center[-1]
    time_i += tupd
    step += 1
    print(f"Time: {time_i:.3f} Gyr, Bound particles: {n_bound}, Total energy: {np.sum(E_var):.3e} (snapshot saved: {step % snapshot_interval == 0})")

mc_xv=np.array(mc_xv)
mc_t=np.array(mc_t)

fname = f'/suphys/aani0116/Work/minicluster_tidal/hpc/mc10_m{mc.mass:.1e}_c{mc.concentration}_d{mc.delta}.h5'

with h5py.File(fname, 'w') as f:
    f.create_dataset('mcp_xv', data=np.array(mcp_xv_snapshots, dtype=np.float64))

    f.create_dataset('bound',    data=np.array(bound_snapshots))
    f.create_dataset('E_p',      data=np.array(E_p_snapshots))
    f.create_dataset('mc_xv',    data=np.array(mc_xv_snapshots))
    f.create_dataset('snap_times', data=np.array(snap_times))
    f.attrs['mcp_mass']        = mcp_mass[0]
    f.attrs['Nbody']          = Nbody
    f.attrs['simulation_time'] = simulation_time
    f.attrs['tupd']            = tupd
    f.attrs['N_orbits']          = N_orbits
    f.attrs['mc_mass']          = mc.mass
    f.attrs['mc_delta']        = mc.delta
    f.attrs['mc_radius']        = mc.radius
    f.attrs['mc_radius_char']   = mc.radius_char
    f.attrs['mc_density_char']  = mc.density_char
    f.attrs['mc_concentration'] = mc.concentration



"""with h5py.File('Work/minicluster_tidal/Tidal_stream/hpc/mc.h5', 'w') as f:
        f.create_dataset('mcp_xv', data=mcp_xv)
        f.create_dataset('mxp_xv_initial', data=mxp_xv_initial)
        f.create_dataset('mc_xv',  data=mc_xv)
        f.create_dataset('mc_t',   data=mc_t)
        f.create_dataset('E_p',   data=E_p)
        f.attrs['mcp_mass'] = mcp_mass
        f.attrs['mc_mass'] = mc.mass
        f.attrs['mc_radius'] = mc.radius
        f.attrs['mc_radius_char'] = mc.radius_char
        f.attrs['mc_density'] = mc.density
        f.attrs['mc_density_char'] = mc.density_char
        f.attrs['mc_concentration'] = mc.concentration"""
