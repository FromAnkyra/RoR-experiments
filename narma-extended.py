"""looks at narma in the context of the Restricted ESN, particularly in terms of sparsity."""

"""runs NARMA on various different types of ESN and restricted ESN"""
from distutils.log import error
import matplotlib.pyplot as plt
import numpy as np
import NymphESN.nymphesn as nymph
import NymphESN.errorfuncs as errorfunc
import NymphESN.restrictedmatrix as rm
import pandas as pd
import seaborn as sns

def get_u(T, seed=None):
    # random stream of inputs u in range 0,0.5 in col 0, 1s (bias) in col 1
    if seed:
        np.random.seed(seed)
    return np.random.uniform(0.0, 0.5, T)


def narmafun(y, u, alpha, beta, gamma, delta):
    # y =[y(t-N+1, ..., y(t-1), y(t))], similar for u
    return alpha * y[-1] + beta * y[-1] * sum(y) + gamma * u[0] * u[-1] + delta


def run_narma(NARMA, T, u, debug=False):
    narmaparams = {
        5: (0.3, 0.05, 1.5, 0.2),  
        10: (0.3, 0.05, 1.5, 0.1),
        20: (0.25, 0.05, 1.5, 0.01),
        30: (0.2, 0.04, 1.5, 0.001)
    }

    # initial NARMA values of y
    for t in range(0, NARMA):
        y = [0] * NARMA

    for t in range(NARMA - 1, T - 1):
        
        y_Nt = [y[i] for i in range(t-NARMA+1, t)]
        u_Nt = [u[i] for i in range(t-NARMA+1, t)]
        y_t1 = narmafun(y_Nt, u_Nt, *narmaparams[NARMA])  # y(t+1) = f(y(t), u(t), ...)
        y.append(y_t1)
    if(debug):
        print('u =', u)
        print('y =', y)
    return [0] + y


# def get_narma_output(shift):
#     vtarget = system['y'].tolist()
#     if shift:
#         # shift forward by one timestep: vtarget(t+1) = y(t)
#         vtarget = [0] + vtarget[:-1]
#     return vtarget

# =========================================================================================================
# =========================================================================================================

rho = 2
density = 0.2
N = 100
rN = 50
NR = 2

mc_threshold = 0.1
TWashout = 1000
TTrain = 500
TTest = 1000
TTot = TWashout + TTrain + TTest

TRuns = 50

NARMA = 5
# run NARMA on
# Part 1:

# First experiment: use density fo 0.05 throughout
# Second experiment: same densities, but use four subreservoirs
# Third experiment: same densities, but use five subreservoirs

# Part 2: 
# Scan through densities to find the optimal density

errordf = pd.DataFrame(columns=['base train', 'base test'])

for i in range(TRuns):
    print(i)
    #create reservoirs
    f = np.tanh
    S_0_2 = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2)
    W_S_0_2 = rm.create_random_esn_weights(N, density=0.2)

    S_0_0_5 = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2)
    W_S_0_0_5 = rm.create_random_esn_weights(N, density=0.05)

    restricted_case_4 = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2)
    W_restricted_4 = rm.create_restricted_esn_weights(N, 25, 4)

    restricted_case_5 = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2)
    W_restricted_5 = rm.create_restricted_esn_weights(N, 20, 5)
    # create weight matrices

    input = get_u(T=TTot, seed=i)
    vtarget = run_narma(NARMA, TTot, input, debug=False)
    # vtarget = np.hstack((np.array([0, 0, 0]), input[:-3]))
    vtarget_np = np.array(vtarget)

    S_0_2.set_data_lengths(TWashout, TTrain, TTest)
    S_0_0_5.set_data_lengths(TWashout, TTrain, TTest)
    restricted_case_4.set_data_lengths(TWashout, TTrain, TTest)
    restricted_case_5.set_data_lengths(TWashout, TTrain, TTest)

    S_0_2.set_input_stream(input)
    S_0_0_5.set_input_stream(input)
    restricted_case_4.set_input_stream(input)
    restricted_case_5.set_input_stream(input)

    S_0_2.run_full(W=[W_S_0_2])
    S_0_0_5.run_full(W=[W_S_0_0_5])
    restricted_case_4.run_full(W=[W_restricted_4])
    restricted_case_5.run_full(W=[W_restricted_5])

    S_0_2.train_reservoir(vtarget_np[TWashout:TWashout+TTrain])
    S_0_0_5.train_reservoir(vtarget_np[TWashout:TWashout+TTrain])
    restricted_case_4.train_reservoir(vtarget_np[TWashout:TWashout+TTrain])
    restricted_case_5.train_reservoir(vtarget_np[TWashout:TWashout+TTrain])

    S_0_2.get_output()
    S_0_0_5.get_output()
    restricted_case_4.get_output()
    restricted_case_5.get_output()

    training02, testing02 = S_0_2.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
    training005, testing005 = S_0_0_5.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
    training4, testing4 = restricted_case_4.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
    training5, testing5 = restricted_case_5.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse) 
    
    errordf.at[i, 's=0.2 train'] = training02
    errordf.at[i, 's=0.2 test'] = testing02
    errordf.at[i, 's=0.05 train'] = training005
    errordf.at[i, 's=0.05 test'] = testing005
    errordf.at[i, 'rN=4 train'] = training4
    errordf.at[i, 'rN=4 test'] = testing4
    errordf.at[i, 'rN=5 train'] = training5
    errordf.at[i, 'rN=5 test'] = testing5
    # # print(i)
    
#plot errors

sns.set_context(rc={'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 18})
fig, ax = plt.subplots(1, 1, figsize =(5, 5))
axb = ax
sns.boxplot(data=errordf, ax=axb, notch=True, width=0.6, linewidth=0.5, fliersize=0)
axb.set_ylabel('NRMSE')
axb.spines['right'].set_visible(False)
axb.spines['top'].set_visible(False)
# axb.set_ylim(-0.3, 2)

plt.show()
#fig.savefig(fn, bbox_inches='tight')

