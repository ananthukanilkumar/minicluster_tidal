import numpy as np
import agama
import h5py
import matplotlib.pyplot as plt
import sys
sys.path.append('/suphys/aani0116/Work/minicluster_tidal/Tidal_stream')
sys.path.append('/suphys/aani0116/Work/minicluster_tidal/stellar')
sys.path.append('/suphys/aani0116/Work/minicluster_tidal/my_modules')
from mc_declare import *
from reverse_orbit import reverse_orbit
agama.setUnits(length=1, velocity=1, mass=1)  # kpc, km/s, Solar mass


pot_mw = agama.Potential('/suphys/aani0116/minicluster/minicluster_tidal/Agama/data/McMillan17.ini')


    

mc = Minicluster(mass=1e-10, delta=10, concentration=10)
Nbody=100000
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




pot_mc = agama.Potential(type='Multipole', particles=(mcp_xv[:, :3], mcp_mass), symmetry='n', rmin=0, rmax=0, gridSizeR=100, lmax=0)


solar_pos = np.array([8.0, 0.0, 0.0, 5.0, 5.0, 220.0]) 

orbit_time = 13

mc_center=reverse_orbit(solar_pos, pot_mw,orbit_time,4, 10)
mc_center_start=mc_center.copy()

mcp_xv += mc_center 

#simulation parameters
simulation_time = orbit_time
tupd = 1e-10 # Gyr
time_i = 0


#stuff for saving data
n_snapshots = 10
snapshot_times = np.linspace(0, simulation_time, n_snapshots)
next_snap = 0

mcp_xv_snapshots=[]
bound_snapshots, E_p_snapshots, mc_xv_snapshots, snap_times = [], [], [], []
n_bound_arr = []


print(tupd)
while time_i < simulation_time:

    mc_time_center, mc_orbit_center = agama.orbit(ic=mc_center, potential=pot_mw, time=tupd,timestart=time_i, trajsize=10, accuracy=1e-10,verbose=False)

    pot_total = agama.Potential(pot_mw,agama.Potential(potential=pot_mc,center=np.column_stack((mc_time_center, mc_orbit_center))))

    mcp_xv = np.vstack(agama.orbit(ic=mcp_xv, potential=pot_total,time=tupd, timestart=time_i, trajsize=1, accuracy=1e-10,dtype=float,verbose=False)[:, 1])

    rel_pos = mcp_xv[:, :3] - mc_orbit_center[-1][:3]
    rel_vel = mcp_xv[:, 3:6] - mc_orbit_center[-1][3:6]
    rel_xv=mcp_xv-mc_orbit_center[-1]

    E_var=0.5*np.linalg.norm(rel_vel, axis=1)**2+pot_mc.potential(rel_pos)
    bound=E_var<0
    n_bound=np.sum(bound)
    n_bound_arr.append(n_bound)
    if n_bound>32:
        pot_mc = agama.Potential(type='NFW',mass=mc.mass_within_radius(5.4*mc.radius_char),scaleRadius=mc.radius_char,)
    if n_bound<=32:
        pot_mc=agama.Potential(type='Plummer', mass=0.0, scaleRadius=1.0)
        tupd=1e-4
        

    

    if next_snap < n_snapshots and time_i >= snapshot_times[next_snap]:
        mcp_xv_snapshots.append(rel_xv.copy())
        bound_snapshots.append(bound.copy())
        E_p_snapshots.append(E_var.copy())
        mc_xv_snapshots.append(mc_orbit_center[-1].copy())
        snap_times.append(time_i)
        next_snap += 1
        print(f'Snapshot {next_snap} saved at time {time_i:.2f} Gyr with {n_bound} bound particles.')



    mc_center = mc_orbit_center[-1]
    time_i += tupd
#need this to get the final snapshot if the loop ends before the last snapshot time is reached
mcp_xv_snapshots.append(rel_xv.copy())
bound_snapshots.append(bound.copy())
E_p_snapshots.append(E_var.copy())
mc_xv_snapshots.append(mc_orbit_center[-1].copy())
snap_times.append(time_i)



fname = f'/suphys/aani0116/Work/minicluster_tidal/hpc/Sigma_test/data/mc_m{mc.mass:.1e}_13B_c{mc.concentration}.h5'

with h5py.File(fname, 'w') as f:
    f.create_dataset('mcp_xv', data=np.array(mcp_xv_snapshots, dtype=np.float64))

    f.create_dataset('bound',    data=np.array(bound_snapshots))
    f.create_dataset('E_p',      data=np.array(E_p_snapshots))
    f.create_dataset('mc_xv',    data=np.array(mc_xv_snapshots))
    f.create_dataset('snap_times', data=np.array(snap_times))
    f.create_dataset('n_bound', data=np.array(n_bound_arr))
    f.attrs['mcp_mass']        = mcp_mass[0]
    f.attrs['Nbody']          = Nbody
    f.attrs['simulation_time'] = simulation_time
    f.attrs['tupd']            = tupd
    f.attrs['mc_mass']          = mc.mass
    f.attrs['mc_delta']        = mc.delta
    f.attrs['mc_radius']        = mc.radius
    f.attrs['mc_radius_char']   = mc.radius_char
    f.attrs['mc_density_char']  = mc.density_char
    f.attrs['mc_concentration'] = mc.concentration
    f.attrs['ic'] = mc_center_start



