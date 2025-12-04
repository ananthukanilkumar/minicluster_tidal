import numpy as np, agama

agama.setUnits(length=1, velocity=1, mass=1) #kpc,km/s, Solar mass

def reverse_orbit(ic,pot,time,timestart,show_start=False):
    orbit_val=agama.orbit(ic=ic, potential=pot, time=-time, timestart=timestart, trajsize=1,accuracy=1e-10)
    times, trajectory = orbit_val
    if show_start:
        print("Starting position:", trajectory[-1])
    return trajectory[-1]



