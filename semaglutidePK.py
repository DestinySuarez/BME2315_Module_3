
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ===== Product-specific parameters =====
products = {
    'Ozempic': dict(Favailable=0.89,
                    ka=0.029,
                    D=2.0,
                    interval=168,
                    n_doses=8,
                    depot='subcutaneous'),

    'Wegovy': dict(Favailable=0.01,
                   ka=0.0029,
                   D=25,
                   interval=24,
                   n_doses=100,
                   depot='gut (oral)'),

    'Cavalitide': dict(Favailable=0.01,
                   ka=0.006,
                   D=21,
                   interval=24,
                   n_doses=100,
                   depot='gut (oral)'),
}

# Shared model parameters
ke = 0.0039     # 1/h, elimination rate
V = 10.8        # L
MW = 4113.6     # g/mol
Ffree = 0.01    # fraction of semaglutide unbound to albumin

# Output resolution
dt = 0.1        # h, spacing between saved output points


# ===== Differential equations =====
def ODEfunc(t, y, ka):
    Asc, Cc = y

    dAsc = -ka * Asc
    dCc = ka * Asc / V - ke * Cc

    return [dAsc, dCc]


# ===== Simulation using solve_ivp =====
def simulate(D, n_doses, interval, ka, Favailable, dt):

    t_out, Asc_out, Cc_out = [], [], []

    y = np.array([0.0, 0.0])

    n_steps = int(round(interval / dt))
    t_eval = np.linspace(0, interval, n_steps + 1)

    for i in range(n_doses):

        # Add a new dose to the depot
        y[0] += Favailable * D

        t0 = i * interval

        # Solve the ODEs over one dosing interval
        sol = solve_ivp(
            ODEfunc,
            (0, interval),
            y,
            args=(ka,),
            t_eval=t_eval
        )

        if not sol.success:
            raise RuntimeError(sol.message)

        # Save results
        t_out.append(t0 + sol.t)
        Asc_out.append(sol.y[0])
        Cc_out.append(sol.y[1])

        # Carry the final state into the next dosing interval
        y = sol.y[:, -1].copy()

    # Combine all dosing intervals
    t = np.concatenate(t_out)
    Asc_out = np.concatenate(Asc_out)
    Cc_out = np.concatenate(Cc_out)

    # Convert mg/L to nM
    Cc_nM = Cc_out * 1e-3 / MW * 1e9

    return t, Asc_out, Cc_nM


# ===== Run both simulations =====
results = {}

for drug, p in products.items():

    t, Asc, Cc = simulate(
        p['D'],
        p['n_doses'],
        p['interval'],
        p['ka'],
        p['Favailable'],
        dt
    )

    results[drug] = (t, Asc, Cc)


# ===== Plot comparison =====
fig, axes = plt.subplots(2, 1, figsize=(10, 9))

for ax, drug in zip(axes, ['Ozempic', 'Wegovy']):

    t, Asc, Cc = results[drug]
    p = products[drug]

    ax.plot(t / 24, Asc,
            label=f"{p['depot']} depot [mg]")

    ax.plot(t / 24, Ffree * Cc,
            label="free in blood [nM]")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Semaglutide")
    ax.set_title(
        f"{drug}: {p['D']} mg every {p['interval']} h"
    )

    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ===== Plot Cavalitide =====

t_cav, Asc_cav, Cc_cav = results['Cavalitide']

plt.figure(figsize=(10, 5))

plt.plot(t_cav / 24, Asc_cav,
         label="gut (oral) depot [mg]")

plt.plot(t_cav / 24, Ffree * Cc_cav,
         label="free in blood [nM]")

plt.xlabel("Time (days)")
plt.ylabel("Cavalitide")
plt.title("Cavalitide: 21 mg every 24 h, 100 doses")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
