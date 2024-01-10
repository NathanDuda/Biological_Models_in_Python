
import numpy as np
import matplotlib.pyplot as plt

def euler_method(sir_t,dsir,dt):
    sir_t_plus_dt = sir_t + dt*dsir 
    return sir_t_plus_dt


def deterministic_SIR_model(alpha, gamma, s, i, N): # no r 
    dsdt = -alpha*s*(i/N)            # i / N because of alpha 
    didt = alpha*s*(i/N) - gamma*i   # i / N because of alpha 
    drdt = gamma*i
    
    return np.array([dsdt,didt,drdt]) # this is sir_t

def stochastic_SIR_model(alpha, gamma, n, s, i, r, dt):

    probability_i = alpha * dt * i/n 
    probability_r = dt * gamma

    new_i = np.random.binomial(s, probability_i)
    new_r = np.random.binomial(i, probability_r)

    s -= new_i
    i += new_i - new_r
    r += new_r

    return np.array([s, i, r])


def main():

    # define constants 
    N = 1000
    alpha = 0.3
    gamma = 0.1
    I0 = 1
    days = 160
    dt = 1.0

    # define initial conditions
    i_deterministic = I0
    s_deterministic = N - i_deterministic
    r_deterministic = 0 

    i_stochastic = I0
    s_stochastic = N - i_stochastic
    r_stochastic = 0 

    # create empty lists
    stochastic_sir = [[],[],[]]
    deterministic_sir = [[],[],[]]

    
    for day in range(days): # assuming dt is 1 

        ########## deterministic 

        # get first derivative 
        dsir_deterministic = deterministic_SIR_model(alpha, gamma, s_deterministic, i_deterministic, N)
        sir_t = np.array([s_deterministic, i_deterministic, r_deterministic])

        # take Euler step 
        s_deterministic, i_deterministic, r_deterministic = euler_method(sir_t,dsir_deterministic,dt)

        # append values to lists
        deterministic_sir[0].append(s_deterministic)
        deterministic_sir[1].append(i_deterministic)
        deterministic_sir[2].append(r_deterministic)


        ########## stochastic 

        # get first derivative 
        s_stochastic, i_stochastic, r_stochastic = stochastic_SIR_model(alpha, gamma, N, s_stochastic, i_stochastic, r_stochastic, dt)

        # append values to lists
        stochastic_sir[0].append(s_stochastic)
        stochastic_sir[1].append(i_stochastic)
        stochastic_sir[2].append(r_stochastic)

    
    # plot
    plt.plot(deterministic_sir[0], label = 'S')
    plt.plot(deterministic_sir[1], label = 'I')
    plt.plot(deterministic_sir[2], label = 'R')
    plt.legend()
    plt.title('Deterministic', loc='center')
    plt.show()

    plt.plot(stochastic_sir[0], label = 'S')
    plt.plot(stochastic_sir[1], label = 'I')
    plt.plot(stochastic_sir[2], label = 'R')
    plt.legend()    
    plt.title('Stochastic', loc='center')
    plt.show() # sometimes flat plot because all infected recover very early 



main()

