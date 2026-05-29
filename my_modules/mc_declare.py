import numpy as np
import agama
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

    def potential(self, random_radius):
        return -agama.G*self.mass_within_radius(random_radius)/random_radius

