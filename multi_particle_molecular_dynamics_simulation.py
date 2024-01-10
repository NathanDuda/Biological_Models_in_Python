

import numpy as np

import matplotlib.pyplot as plt


# function to calculate the force and potential energy: 
def calc_force(position, equilibrium_distance, k):
    force = -k * (position - equilibrium_distance)
    potential_energy = 1/2 * k * np.sum((position - equilibrium_distance) ** 2)
    return force, potential_energy


# function for a single update using the Verlet algorithm 
def verlet(position_2,position_1, force, m, dt):
    position_3 = (2 * position_2) - position_1 + (((dt ** 2) / m) * force)
    return position_3


# initialize position
def init_position(num_particles):
    position_1 = np.zeros((num_particles,3))

    for i in range(num_particles):
        p = np.random.rand(3)
        position_1[i] = p

    return position_1


# function to initialize velocity 
def init_velocity(k,T,m):
    std_dev = (k * T / m) ** 0.5
    velocity_1 = np.random.normal(0, std_dev, 1)
    return velocity_1


# function to take on Euler step 
def Euler_step(position_1,dt,m,force,velocity_1):
    position_2 = position_1 + (dt * velocity_1) + (((dt ** 2)/(2 * m)) * force)
    return position_2


# function to plot energies as a function of time
def plot_energies(potential_energies,kinetic_energies,total_energies,times):
    plt.plot(potential_energies, label='Potential Energy')
    plt.plot(kinetic_energies, label='Kinetic Energy')
    plt.plot(total_energies, label='Total Energy')
    plt.legend()
    plt.show()


# calculate force and potential energy for multi-particle system:
def lennard_jones_force_and_energy(positions,sigma,epsilon):
    num_particles = positions.shape[0]
    forces = np.zeros((num_particles,3),dtype=np.float64)
    total_potential_energy = 0.0

    for i in range(num_particles):
        for j in range(i+1, num_particles):
            rij = positions[j] - positions[i]
            r = np.linalg.norm(rij)

            # implement spherical cuttoff:
            if (r < r_c): # use only particles closer than r_c
                Force_mag = 24 * epsilon * (2*(sigma/r)**13 - (sigma/r)**7) / r
                Force_dir = Force_mag * rij / r

                total_potential_energy += 4 * epsilon * ((sigma / r) ** 13 - (sigma / r) ** 7)

                forces[i] = forces[i] - np.array(Force_dir) # force of i on j is negative of j on i (Newtons laws)
                forces[j] = forces[j] + np.array(Force_dir)

    return forces, total_potential_energy




# main function 
def main():

    # define constants 
    k = 1.0
    m = 1.0
    T = 0.0001
    equilibrium_distance = np.array([0.0,0.0,0.0])

    dt = 0.1
    num_time_steps = 100

    num_particles = 10
    r_c=5

    sigma = 1.0
    epsilon = 4.0

    # initialization of position
    position_1 = init_position(1) 
    #position_1 = init_position(num_particles) # for-multi-particle


    # initialization of velocity
    velocity_1 = init_velocity(k,T,m)
    #velocity_1 = np.array([init_velocity(k,m,T) for i in range(num_particles)]) # for-multi-particle


    # initialization of forces
    force, potential_energy = calc_force(position_1, equilibrium_distance, k)
    #force, potential_energy = lennard_jones_force_and_energy(position_1, sigma, epsilon) # for-multi-particle


    # one step of Euler's method 
    position_2 = Euler_step(position_1,dt,m,force,velocity_1)

    # create empty lists
    positions = []
    potential_energies = []
    kinetic_energies = []
    total_energies = []
    times = []

    # for loop
    for i in range(1,num_time_steps + 1):

        # position update
        position_3 = verlet(position_2, position_1, force, m, dt)

        # energies updates
        velocity = (1/(2 * dt) * (position_3 - position_1))
        kinetic_energy = (1/2) * m * (velocity ** 2) 
        kinetic_energy = kinetic_energy[0] + kinetic_energy[1] + kinetic_energy[2]

        # calculate total energy 
        total_energy = potential_energy + kinetic_energy

        # force update 
        force, potential_energy = calc_force(position_3, equilibrium_distance, k)


        positions.append(position_3)
        potential_energies.append(potential_energy)
        kinetic_energies.append(kinetic_energy)
        total_energies.append(total_energy)

        position_1 = position_2
        position_2 = position_3

    # plot energies as a function of time
    plot_energies(potential_energies,kinetic_energies,total_energies,times)

    

main()


