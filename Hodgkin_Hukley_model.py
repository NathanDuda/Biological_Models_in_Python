import numpy as np
import matplotlib.pyplot as plt

class BioModelBase:
    def __init__(self, params, initial_state,dt):
        self.params = params
        self.states = [np.array(initial_state)]
        self.dt = dt

    def current_state(self):
        # Return the latest state
        return self.states[-1]

    def evaluate(self, total_time):
        steps = int(total_time / self.dt)
        for _ in range(steps):
            self.integrate()

    def step(self):
        # This needs to be implemented in a subclass
        raise NotImplementedError("The step method must be implemented.")

    def integrate(self):
        current = self.current_state()
        dydt = self.step()
        new_state = current + dydt * self.dt
        self.states.append(new_state)
        return new_state

    def plot(self):
        # This needs to be implemented in a subclass
        raise NotImplementedError("The plot method must be implemented.")


###############################################################


class HodgkinHuxleyModel(BioModelBase):
    def __init__(self,params,initial_state,dt):
        super().__init__(params,initial_state,dt)
    
    def alpha_n(self, V):
        return 0.01 * (V + 55) / (1 - np.exp(-0.1 * (V + 55)))
    def beta_n(self, V):
        return 0.125 * np.exp(-0.0125 * (V + 65))
    def alpha_m(self, V):
        return 0.1 * (V + 40) / (1 - np.exp(-0.1 * (V + 40)))
    def beta_m(self, V):
        return 4 * np.exp(-0.0556 * (V + 65))
    def alpha_h(self, V):
        return 0.07 * np.exp(-0.05 * (V + 65))
    def beta_h(self, V):
        return 1 / (1 + np.exp(-0.1 * (V + 35)))
    
    def step(self):
        # get current values 
        V, n, m, h  = self.current_state()  
        gNa, gK = self.params['gNa'], self.params['gK']
        ENa, EK = self.params['ENa'], self.params['EK']
        I = self.params['I']

        # get alpha and beta values for n, m, and h each 
        alpha_n = self.alpha_n(V)
        beta_n  = self.beta_n(V)
        alpha_m = self.alpha_m(V)
        beta_m  = self.beta_m(V)
        alpha_h = self.alpha_h(V)
        beta_h  = self.beta_h(V)

        # get the derivatives of n, m, and h using the three equations 
        dn = alpha_n * (1 - n) - beta_n * n # change in n (Na channel activation)
        dm = alpha_m * (1 - m) - beta_m * m
        dh = alpha_h * (1 - h) - beta_h * h

        # get current for each channel
        INa = gNa * m**3 * h * (V - ENa) # sodium current
        IK = gK * n**4 * (V - EK) # potassium current

        # get the rate of change (derivative) of voltage 
        dV = I - INa - IK # subtract the currents in the channels from the applied current

        return np.array([dV, dn, dm, dh])

    def plot(self):
        # get the x axis (time) for the plot 
        time_points = np.arange(0, len(self.states) * self.dt, self.dt)

        # save the V values for the action potential plot 
        V_values = [state[0] for state in self.states]

        # save the n, m, and h, values for the gating activation/inactivation plot 
        n_values = [state[1] for state in self.states]
        m_values = [state[2] for state in self.states]
        h_values = [state[3] for state in self.states]

        # plot the action potential 
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(time_points, V_values, label='Membrane Potential (V)')
        plt.xlabel('Time (ms)')
        plt.ylabel('mV')
        plt.legend()
        plt.grid(True)

        # plot the gating activation/inactivation over time 
        plt.subplot(2, 1, 2)
        plt.plot(time_points, n_values, label='n (Na activation)')
        plt.plot(time_points, m_values, label='m (K  activation)')
        plt.plot(time_points, h_values, label='h (Na inactivation)')
        plt.xlabel('Time (ms)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


# Parameters:
params = {
    'gNa': 120, 'gK': 36, 'gL': 0.3,
    'ENa': 50, 'EK': -77, 'EL': -54.387, 
    'I': 10}
# Time step
dt = 0.01 # ms
# Initial state: [membrane potential, n, m, h]
initial_state = [-65, 0.3177, 0.0529, 0.5961]


# run for 70 ms
hodgkin_huxley_model = HodgkinHuxleyModel(params, initial_state, dt)
hodgkin_huxley_model.evaluate(70)
# plot 
hodgkin_huxley_model.plot()

