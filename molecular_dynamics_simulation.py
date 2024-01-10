

import numpy as np

import matplotlib.pyplot as plt


# function to calculate the force and potential energy: 
def calc_force(position, equilibrium_distance, k):
    force_x = -k * (position[0] - equilibrium_distance[0])
    force_y = -k * (position[1] - equilibrium_distance[1])
    force_z = -k * (position[2] - equilibrium_distance[2])

    force = np.array([force_x,force_y,force_z])

    potential_energy_x = 1/2 * k * (position[0] - equilibrium_distance[0]) ** 2
    potential_energy_y = 1/2 * k * (position[1] - equilibrium_distance[1]) ** 2
    potential_energy_z = 1/2 * k * (position[2] - equilibrium_distance[2]) ** 2

    potential_energy = ((potential_energy_x **2) + (potential_energy_y ** 2) + (potential_energy_z ** 2)) ** 0.5
    #potential_energy = np.array([potential_energy_x,potential_energy_y,potential_energy_z])

    return force, potential_energy


# function for a single update using the Verlet algorithm 
def verlet(position_2,position_1, force, m, dt):

    r_x = (2 * position_2[0]) - position_1[0] + (((dt ** 2) / m) * force[0])
    r_y = (2 * position_2[1]) - position_1[1] + (((dt ** 2) / m) * force[1])
    r_z = (2 * position_2[2]) - position_1[2] + (((dt ** 2) / m) * force[2])

    position_3 = [r_x,r_y,r_z] # make list
    position_3 = np.asarray(position_3)

    return position_3


# function to initialize velocity 
def init_velocity(k,T,m):
    velocity_1 = np.zeros(3)

    v = (3/2) * k * T

    velocity_1[0] = np.random.normal(v, v, 1)
    velocity_1[1] = np.random.normal(v, v, 1)
    velocity_1[2] = np.random.normal(v, v, 1)
    
    return velocity_1


# function to take on Euler step 
def Euler_step(position_1,dt,m,force,velocity_1): # I made this up 
    position_2 = np.zeros(3)
    position_2[0] = position_1[0] + (dt * velocity_1[0]) + (((dt ** 2)/(2 * m)) * force[0])
    position_2[1] = position_1[1] + (dt * velocity_1[1]) + (((dt ** 2)/(2 * m)) * force[1])
    position_2[2] = position_1[2] + (dt * velocity_1[2]) + (((dt ** 2)/(2 * m)) * force[2])
    return position_2


# function to plot energies as a function of time
def plot_energies(potential_energies,kinetic_energies,total_energies,times):
    plt.plot(potential_energies, label='Potential Energy')
    plt.plot(kinetic_energies, label='Kinetic Energy')
    plt.plot(total_energies, label='Total Energy')
    plt.legend()
    plt.show()


# main function 
def main():

    # define constants 
    k = 1.0
    m = 1.0
    T = 0.0001
    force_constant = 1.0 # not used 
    equilibrium_distance = np.array([0.0,0.0,0.0])

    dt = 0.1
    num_time_steps = 100

    # initialization of position
    position_1 = np.random.uniform(low=0,high=1,size=3) # includes low but excludes high 

    # initialization of velocity
    velocity_1 = init_velocity(k,T,m)

    # initialization of forces
    force, potential_energy = calc_force(position_1, equilibrium_distance, k)
    
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
        kinetic_energy = ((kinetic_energy[0] ** 2) + (kinetic_energy[1] ** 2) + (kinetic_energy[2] ** 2)) ** 0.5

        #print(position_3 - position_1)
        total_energy = potential_energy + kinetic_energy

        # force update 
        force, potential_energy = calc_force(position_3, equilibrium_distance, k)


        positions.append(position_3)
        potential_energies.append(potential_energy)
        kinetic_energies.append(kinetic_energy)
        total_energies.append(total_energy)
        times.append(dt)

        position_1 = position_2
        position_2 = position_3

    # plot energies as a function of time
    plot_energies(potential_energies,kinetic_energies,total_energies,times)

    

main()



#################################################

# additional function available to test accuracy 

def test_verlet(initial_time,dt,force,m,k):
    omega = (k/m) ** 0.5

    time = initial_time
    position_1 = np.cos(omega * time)

    time = initial_time + dt
    position_2 = np.cos(omega * time)

    time = initial_time + 2 * dt
    position_3 = np.cos(omega * time)

    verlet_position = verlet(position_2,position_1, force, m, dt)

    unsigned_error = abs(position_3 - verlet_position)

    return unsigned_error


