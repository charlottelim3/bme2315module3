# cavalitidePKPD.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Fact model ------------------------------------------------------------------
def Fact(L1, L2, Kd1, Kd2):
    """Fractional receptor activation with two agonists L1 and L2"""
    return (Kd2*L1+Kd1*L2) / (Kd1*Kd2 + Kd2*L1 + Kd1*L2)

# PD parameters [nM] ------------------------------------------------------------------
Kd1 = 0.17      # GLP1
Kd2 = 0.077     # Cavalitide (optimized fit from Task 1)
L1 = 0.015      # GLP1 (fasting)

# dosing regimens: initial (0.5 nM) and adjusted (~90% active receptors) ------------------------------------------------------------------
products = {
    'initial': dict(Favailable=0.01,   # bioavailability
                    ka=0.006,          # 1/h, absorption rate
                    D=20.75,            # mg per dose
                    interval=24,       # h between doses (daily)
                    n_doses=100),      # number of doses (100 days)
    'adjusted': dict(Favailable=0.01,
                    ka=0.006,
                    D=28.5,
                    interval=24,
                    n_doses=100),
}

# shared PK model parameters ------------------------------------------------------------------
ke = 0.0039     # 1/h, elimination rate
V = 10.8        # L
MW = 4113.6     # g/mol
Ffree = 0.01    # fraction free in blood (rest bound to albumin)

# simulation parameters ------------------------------------------------------------------
dt = 1.0        # h, spacing of saved output points

def simulate(drug):
    """Repeated dosing every `interval` h; returns time (h), free blood
    concentration (nM) and fraction of active receptors."""
    p = products[drug]
    Favailable, ka, D = p['Favailable'], p['ka'], p['D']
    interval, n_doses = p['interval'], p['n_doses']

    # PK ODEs: Asc = depot amount (mg), Cc = blood concentration (mg/L)
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
        sol = solve_ivp(ODEfunc, [0, interval], y, t_eval=t_eval,
                        rtol=1e-8, atol=1e-10)   # solve one dosing interval
        t_out.append(i*interval + sol.t)
        Cc_out.append(sol.y[1])
        y = sol.y[:, -1].copy()                  # carry state into next interval
    t = np.concatenate(t_out)
    Cc = np.concatenate(Cc_out)
    Cc_nM = Ffree * Cc * 1e-3 / MW * 1e9         # mg/L -> free nM
    Fact_t = Fact(L1, Cc_nM, Kd1, Kd2)           # PK output feeds the PD model
    return t, Cc_nM, Fact_t

# Simulations ------------------------------------------------------------------
results = {drug: simulate(drug) for drug in products}

# Initial dosing plot, Fact over 100 days ------------------------------------------------------------------
plt.figure()
t, Cc, Fact_t = results['initial']
plt.plot(t / 24, Fact_t, label="Initial")
plt.ylim(0, 1.05)
plt.xlabel("Time (days)")
plt.ylabel("Fraction active (Fact)")
plt.title(f"Cavalitide: {products['initial']['D']} mg every 24 h")
plt.legend()
plt.grid(alpha=0.3)

# Adjusted dosing plot w/ 90% target ---------------------------------------------------------------------
plt.figure()
t, Cc, Fact_t = results['adjusted']
plt.plot(t / 24, Fact_t, label="Adjusted Dosage")
plt.axhline(0.9, color="k", ls="--", label="90% target")
plt.ylim(0, 1.05)
plt.xlabel("Time (days)")
plt.ylabel("Fraction active (Fact)")
plt.title(f"Cavalitide: {products['adjusted']['D']} mg every 24 h")
plt.legend()
plt.grid(alpha=0.3)

plt.show()

# Steady-state (last dosing interval) values ------------------------------------------------------------------
for drug in products:
    t, Cc, Fact_t = results[drug]
    print(f"{drug}: free Cavalitide = {Cc[-25:].mean():.3f} nM, Fact = {Fact_t[-25:].mean():.3f}")