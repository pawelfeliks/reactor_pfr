import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

#######################################################################
# Input Parameters
#######################################################################

T_0 = 1500.0  # inlet temperature [K]
pressure = ct.one_atm  # constant pressure [Pa]
composition_0 = 'H2:2, O2:1, AR:0.1'
length = 1.5e-7  # *approximate* PFR length [m]
u_0 = .006  # inflow velocity [m/s]
area = 1.e-4  # cross-sectional area [m**2]

# input file containing the reaction mechanism
reaction_mechanism = 'h2o2.yaml'

# Resolution: The PFR will be simulated by 'n_steps' time steps or by a chain
# of 'n_steps' stirred reactors.
n_steps = 2000
#####################################################################


#####################################################################
# Method 1: Lagrangian Particle Simulation
#####################################################################
# A Lagrangian particle is considered which travels through the PFR. Its
# state change is computed by upwind time stepping. The PFR result is produced
# by transforming the temporal resolution into spatial locations.
# The spatial discretization is therefore not provided a priori but is instead
# a result of the transformation.

# import the gas model and set the initial conditions
gas1 = ct.Solution(reaction_mechanism)
gas1.TPX = T_0, pressure, composition_0
mass_flow_rate1 = u_0 * gas1.density * area

# create a new reactor
r1 = ct.IdealGasConstPressureReactor(gas1)
# create a reactor network for performing time integration
sim1 = ct.ReactorNet([r1])

# approximate a time step to achieve a similar resolution as in the next method
t_total = length / u_0
dt = t_total / n_steps
# define time, space, and other information vectors
t1 = (np.arange(n_steps) + 1) * dt
z1 = np.zeros_like(t1)
u1 = np.zeros_like(t1)
states1 = ct.SolutionArray(r1.thermo)
for n1, t_i in enumerate(t1):
    # perform time integration
    sim1.advance(t_i)
    # compute velocity and transform into space
    u1[n1] = mass_flow_rate1 / area / r1.thermo.density
    z1[n1] = z1[n1 - 1] + u1[n1] * dt
    states1.append(r1.thermo.state)
#####################################################################

plt.figure()
plt.plot(z1, states1.T, label='Lagrangian Particle')
plt.xlabel('$z$ [m]')
plt.ylabel('$T$ [K]')
plt.legend(loc=0)
plt.show()
plt.savefig('pfr_T_z.png')

plt.figure()
plt.plot(t1, states1.X[:, gas1.species_index('H2')], label='H2')
plt.xlabel('$t$ [s]')
plt.ylabel('$X_{H_2}$ [-]')
plt.legend(loc=0)
plt.show()
plt.savefig('pfr_XH2_t.png')

plt.clf()
plt.subplot(2, 2, 1)
plt.plot(t1, states1.T)
plt.xlabel('Time (ms)')
plt.ylabel('Temperature (K)')
plt.subplot(2, 2, 2)
plt.plot(t1, states1.X[:, gas1.species_index('H2')])
plt.xlabel('Time (ms)')
plt.ylabel('OH Mole Fraction')
plt.subplot(2, 2, 3)
plt.plot(t1, states1.X[:, gas1.species_index('H')])
plt.xlabel('Time (ms)')
plt.ylabel('H Mole Fraction')
plt.subplot(2, 2, 4)
plt.plot(t1, states1.X[:, gas1.species_index('H2')])
plt.xlabel('Time (ms)')
plt.ylabel('H2 Mole Fraction')
plt.tight_layout()
plt.show()

