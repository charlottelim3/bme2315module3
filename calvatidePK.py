# cavalitidePK.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# product-specific parameters
products = {
    'Ozempic': dict(Favailable=0.89,   # bioavailability
                    ka=0.029,          # 1/h, absorption rate
                    D=2.0,             # mg per dose
                    interval=168,      # h between doses (weekly)
                    n_doses=8,         # number of doses
                    depot='subcutaneous'),
    'Wegovy':  dict(Favailable=0.01,
                    ka=0.0029,
                    D=25,              # mg
                    interval=24,       # h between doses (daily)
                    n_doses=100,
                    depot='gut (oral)'),
    # Task 2b: Cavalitide daily pill with increased absorption rate
    'Cavalitide': dict(Favailable=0.01,
                       ka=0.006,
                       D=25,
                       interval=24,
                       n_doses=100,
                       depot='gut (oral)'),
    # Task 2c: Cavalitide with dose lowered to give ~0.5 nM steady state
    'Cavalitide (optimized)': dict(Favailable=0.01,
                                   ka=0.006,
                                   D=20.8,
                                   interval=24,
                                   n_doses=100,
                                   depot='gut (oral)'),
}

# shared model parameters
ke = 0.0039     # 1/h, elimination rate
V = 10.8        # L
MW = 4113.6     # g/mol
Ffree = 0.01    # fraction free in blood (rest bound to albumin)

# simulation parameters
dt = 1.0        # h, spacing of saved output points

def simulate(drug):
    """Repeated dosing every `interval` h; returns time (h) and
    free blood concentration (nM)."""
    p = products[drug]
    Favailable, ka, D = p['Favailable'], p['ka'], p['D']
    interval, n_doses = p['interval'], p['n_doses']

    # ODEs: Asc = depot amount (mg), Cc = blood concentration (mg/L)
    def ODEfunc(t, y):
        Asc, Cc = y
        dAsc = -ka*Asc
        dCc = ka*Asc/V - ke*Cc
        return [dAsc, dCc]

    t_out, Cc_out = [], []
    y = np.array([0.0, 0.0])
    t_eval = np.arange(0, interval + dt/2, dt)   # output times within each interval
    for i in range(n_doses):
        y[0] += Favailable*D                     # absorbed fraction of dose enters the depot
        # solve_ivp replaces the explicit Euler loop for this dosing interval
        sol = solve_ivp(ODEfunc, [0, interval], y, t_eval=t_eval,
                        rtol=1e-8, atol=1e-10)
        t_out.append(i*interval + sol.t)
        Cc_out.append(sol.y[1])
        y = sol.y[:, -1].copy()                  # carry state into next interval
    t = np.concatenate(t_out)
    Cc = np.concatenate(Cc_out)
    Cc_nM = Ffree * Cc * 1e-3 / MW * 1e9         # mg/L -> free nM
    return t, Cc_nM

# --- run all simulations ---
results = {drug: simulate(drug) for drug in products}

# --- Task 2a plot: Ozempic vs Wegovy ---
plt.figure()
for drug in ['Ozempic', 'Wegovy']:
    t, Cc = results[drug]
    plt.plot(t / 24, Cc, label=drug)
plt.xlabel("Time (days)")
plt.ylabel("Free semaglutide in blood [nM]")
plt.title("Ozempic (weekly injection) vs. Wegovy (daily pill)")
plt.legend()
plt.grid(alpha=0.3)

# --- Task 2b plot: Cavalitide blood concentration ---
plt.figure()
t, Cc = results['Cavalitide']
plt.plot(t / 24, Cc, label="Cavalitide")
plt.xlabel("Time (days)")
plt.ylabel("Free Cavalitide in blood [nM]")
plt.title("Cavalitide: 25 mg every 24 h, 100 doses")
plt.legend()
plt.grid(alpha=0.3)

# --- Task 2c plot: optimized Cavalitide dosing vs. 0.5 nM target ---
plt.figure()
t, Cc = results['Cavalitide (optimized)']
plt.plot(t / 24, Cc, label="Cavalitide (optimized)")
plt.axhline(0.5, color="k", ls="--", label="0.5 nM target")
plt.xlabel("Time (days)")
plt.ylabel("Free Cavalitide in blood [nM]")
plt.title("Cavalitide: 20.8 mg every 24 h, 100 doses")
plt.legend()
plt.grid(alpha=0.3)

plt.show()

# steady-state value (last dosing interval) for the optimized dose
t, Cc = results['Cavalitide (optimized)']
print(f"Optimized Cavalitide steady state: {Cc[-25:].min():.3f} - {Cc[-25:].max():.3f} nM")