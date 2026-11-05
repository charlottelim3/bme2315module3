
import numpy as np
import matplotlib.pyplot as plt

# Rename/add parameters to accommodate GLP1 and Semaglutide and their respective dissociation constants
def Fact(GLP1, Semaglutide, Kd_G, Kd_S):
    """Fractional receptor activation by GLP1 and Semaglutide"""
    #Formula derived as seen on Lecture 2
    return (((Kd_S*GLP1) + (Kd_G*Semaglutide)) / ((Kd_G*Kd_S) + (Kd_S*GLP1) + (Kd_G*Semaglutide)))

Kd_G = 0.17 #dissociation constant for GLP1 [nM]
Kd_S = 0.38 #dissociation constant for Semaglutide [nM]
G_fasting = 0.015 # GLP1 concentration in fasting state [nM]
G_ate = 0.040 # GLP1 concentration after feeding[nM]
Semaglutide = np.logspace(-2, 4, 500) # Semaglutide concentration, 0.01 to 10,000 [nM]


plt.figure()
#Edited the plotting function to include both GLP1 fasting and fed states with different colors and labels
plt.semilogx(Semaglutide, Fact(G_fasting, Semaglutide, Kd_G, Kd_S), color = 'blue', lw=2, label = "GLP1 Fasting")
plt.semilogx(Semaglutide, Fact(G_ate, Semaglutide, Kd_G, Kd_S), color = 'orange', lw=2, label = "GLP1 Fed")
plt.axhline(0.5, ls="--", c="gray", lw=1)
plt.ylim(0, 1)
plt.xlabel("[Semaglutide] (nM, log scale)")
plt.ylabel("Fraction active (Fact)")
plt.legend() #Added legend to differentiate between GLP1 fasting and fed states
plt.show()

