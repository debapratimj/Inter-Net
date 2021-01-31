# Initialises interneuron network for futute runs

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter , lfilter
from scipy.fftpack import fft, fftfreq
%matplotlib inline


'''  MODEL PARAMETERS '''

Cm = 100*pF                    # membrane capacitance
g_leak =10*nS                  # leak conductance
E_rest = -65*mvolt             # membrane rest potential
V_thresh = -52*mvolt           # membrane threshold potential
V_reset = -67*mvolt            # membrane reset potential 
tau_m = 10*ms                  # membrane time constant
tau_r = 1*ms                   # absolute refractory period

E_i  = -75*mvolt               # inhib. reversal potential
E_e  =  0*mvolt                # excit. reversal potential

tau_l = 1*ms                   # time delay 
tau_r_e  = 0.5*ms
tau_d_e  = 2*ms
tau_r_i  = 0.45*ms
tau_d_i  = 1.2*ms

g_peak_e = 1*nS
g_peak_i = 5*nS


N_inhib = 200      # number of inhibitory inter neurons in the network
N_excit = 800      # each inter neuron gets input from 800 excit. neurons.


'''   MODEL  '''

def model(sim_dur, inp_freq, tau_d_i=1.2*ms, g_peak_i=5*nS):
    defaultclock.dt = 0.05*ms             # choose a particular time step
    timestep = 0.05*ms
    input_rate = (inp_freq/N_excit)*Hz       # total input to each neuron is 5 KHz
    alpha = 20 /ms

    ''' ----------  NEURON EQUATIONS ------------- '''

    eqs_model = '''
    dv/dt = (g_leak*(E_rest - v) + I_syn )/Cm : volt
    I_syn = I_ampa_exc + I_rec : amp
    I_ampa_exc = g_peak_e*(E_e - v)*s_ampa_tot : amp
    I_rec = g_peak_i*(E_i -v)*s_rec_tot : amp
    s_ampa_tot : 1
    s_rec_tot  : 1
    '''

    eqs_ampa = '''
    s_ampa_tot_post = w * s_ampa : 1 (summed)
    ds_ampa / dt = - s_ampa / tau_d_e + alpha * x * (1 - s_ampa) : 1 (clock-driven)
    dx / dt = - x / tau_r_e : 1 (clock-driven)
    w : 1
    '''
    eqs_pre_ampa = '''
    x += 1
    '''
    eqs_rec = '''
    s_rec_tot_post = w * s_rec : 1 (summed)
    ds_rec / dt = - s_rec / tau_d_i + alpha * y * (1 - s_rec) : 1 (clock-driven)
    dy / dt = - y / tau_r_i : 1 (clock-driven)
    w : 1
    '''
    eqs_pre_rec = '''
    y += 1
    '''
    #clip(gi, 0, g_peak_i)
    P = PoissonGroup(8400, rates=input_rate)

    G = NeuronGroup(N_inhib, eqs_model, threshold='v> V_thresh', reset='v = V_reset', method ='euler')

    # Excitatory Synapse Group
    S_excit = Synapses(P,G, model=eqs_ampa , on_pre= eqs_pre_ampa , delay = tau_l , method = 'euler')
    S_excit.connect(p=0.095)
    S_excit.w = 1.0
    # Inhibitory, Recurrent Synapse Group

    S_inhib = Synapses(G,G,model=eqs_rec , on_pre= eqs_pre_rec , delay = tau_l , method = 'euler')
    S_inhib.connect(condition='i!=j', p=0.2)
    # set inhib weights to 1.0 as well
    S_inhib.w = 1.0
    # Spike, State Monitors

    State_Mon = StateMonitor(G, 'v',record=True)
    Spike_Mon = SpikeMonitor(G)
    Rate_Mon = PopulationRateMonitor(G)

    # initialise at rest 
    G.v = E_rest
    print('Before v = %s' % G.v[0])
    run(sim_dur)
    print('After v = %s' % G.v[0])
    
    return (State_Mon, Spike_Mon, Rate_Mon)
