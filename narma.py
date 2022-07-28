"""runs NARMA on various different types of ESN and restricted ESN"""
from distutils.log import error
import matplotlib.pyplot as plt
import numpy as np
import NymphESN.nymphesn as nymph
import NymphESN.errorfuncs as errorfunc
import NymphESN.restrictedmatrix as rm
import pandas as pd
import seaborn as sns
import scipy.stats as st

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
TTest = 1000

TRuns = 50

NARMA = 5
# run NARMA on
# - an N=100 reservoir? - base_case
# - an N=100 restricted reservoir? restricted_case
# - an N=100 rr w/ 2 timescales on the circuit model circuit_case
# - an N=100 rr w/ 2 timescales on the swarm model swarm_case
# - an N=100 rr w/ 2 timescales on the beacon model beacon_case
def experiment(NARMA, TTrain):    
    errordf = pd.DataFrame(columns=['base train', 'base test'])
    TTot = TWashout + TTrain + TTest
    traindom = 0
    testdom = 0

    for i in range(TRuns):
        # print(i)
        #create reservoirs
        f = np.tanh
        base_case = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2)
        W_base = rm.create_random_esn_weights(N)
        restricted_case = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2)
        W_restricted = rm.create_restricted_esn_weights(N, rN, NR)
        # print(W_restricted)
        # circuit_case = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2)
        # W_circuit = rm.zero_Wn(W_restricted, rN, 1)
        # # print(W_circuit)
        # swarm_case = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2)
        # W_swarm = rm.zero_all_On(W_circuit, rN, 1)
        # W_swarm = rm.zero_On_all(W_swarm, rN, 1)
        # # print(W_swarm)
        # gondor_case = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2)
        # Wu_restricted = rm.create_restricted_esn_input_weights(N, 1)
        # Wu_gondor = rm.zero_Un(Wu_restricted, rN, 1)
        # # print(np.array_equal(W_swarm, W_circuit))
        # W_gondor = rm.zero_On_all(W_circuit, rN, 1)
        # create weight matrices

        input = get_u(T=TTot, seed=i)
        vtarget = run_narma(NARMA, TTot, input, debug=False)
        # vtarget = np.hstack((np.array([0, 0, 0]), input[:-3]))
        vtarget_np = np.array(vtarget)

        base_case.set_data_lengths(TWashout, TTrain, TTest)
        restricted_case.set_data_lengths(TWashout, TTrain, TTest)
        # circuit_case.set_data_lengths(TWashout, TTrain, TTest)
        # swarm_case.set_data_lengths(TWashout, TTrain, TTest)
        # gondor_case.set_data_lengths(TWashout, TTrain, TTest)

        base_case.set_input_stream(input)
        restricted_case.set_input_stream(input)
        # circuit_case.set_input_stream(input)
        # swarm_case.set_input_stream(input)
        # gondor_case.set_input_stream(input)

        base_case.run_full(W=[W_base])
        restricted_case.run_full(W = [W_restricted])
        # circuit_case.run_full(W = [W_restricted, W_circuit])
        # swarm_case.run_full(W = [W_restricted, W_swarm])
        # gondor_case.run_full(W = [W_restricted, W_gondor], Wu = [Wu_restricted, Wu_gondor])

        base_case.train_reservoir(vtarget_np[TWashout:TWashout+TTrain])
        restricted_case.train_reservoir(vtarget_np[TWashout:TWashout+TTrain])
        # circuit_case.train_reservoir(vtarget_np[TWashout:TWashout+TTrain])
        # swarm_case.train_reservoir(vtarget_np[TWashout:TWashout+TTrain])
        # gondor_case.train_reservoir(vtarget_np[TWashout:TWashout+TTrain])

        base_case.get_output()
        restricted_case.get_output()
        # circuit_case.get_output()
        # swarm_case.get_output()
        # gondor_case.get_output()

        btraining, btesting = base_case.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
        rtraining, rtesting = restricted_case.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
        # ctraining, ctesting = circuit_case.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
        # straining, stesting = swarm_case.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
        # gtraining, gtesting = gondor_case.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
        # get_error does not return numbers
        # print(rtraining)
        # print(ctraining)
        # print(straining)
        # print(gtraining)
        errordf.at[i, 'base train'] = btraining
        errordf.at[i, 'base test'] = btesting
        errordf.at[i, 'restr train'] = rtraining
        errordf.at[i, 'restr test'] = rtesting
        # errordf.at[i, 'circ train'] = ctraining
        # errordf.at[i, 'circ test'] = ctesting
        # errordf.at[i, 'swarm train'] = straining
        # errordf.at[i, 'swarm test'] = stesting
        # errordf.at[i, 'gondor train'] = gtraining
        # errordf.at[i, 'gondor test'] = gtesting

        # # print(i)

        #boxplot
        # sns.set_context(rc={'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 18})
        # fig, ax = plt.subplots(1, 1, figsize =(5, 5))
        # axb = ax
        # sns.boxplot(data=errordf, ax=axb, notch=True, width=0.6, linewidth=0.5, fliersize=0)
        # axb.set_ylabel('NRMSE')
        # axb.spines['right'].set_visible(False)
        # axb.spines['top'].set_visible(False)
        # # axb.set_ylim(-0.3, 2)

# plt.show()
#fig.savefig(fn, bbox_inches='tight')
    
    #descriptive stats
    # average = {}
    # for column in errordf:
    #     nmsres = errordf[column].to_numpy()
    #     mean = np.mean(nmsres)
    #     sd = np.std(nmsres)
    #     average[column] = (mean, sd)
    
    #dominance test
    # for i in range(TRuns):
    #     for j in range(TRuns):
    #         if errordf['base train'][i] < errordf['restr train'][j]:
    #             traindom += 1
    #         elif errordf['base train'][i] == errordf['restr train'][j]:
    #             traindom += 0.5
    #         if errordf['base test'][i] < errordf['restr test'][j]:
    #             testdom += 1
    #         elif errordf['base test'][i] == errordf['restr test'][j]:
    #             testdom += 0.5
    
    # traindom = traindom/(TRuns**2)
    # testdom = testdom/(TRuns**2)

    #t-tests
    W_train, train_t = st.brunnermunzel(errordf['base train'].to_numpy(), errordf['restr train'].to_numpy())
    W_test, test_t = st.brunnermunzel(errordf['base test'].to_numpy(), errordf['restr test'].to_numpy())

    return  (W_train, train_t), (W_test, test_t)
    # return traindom, testdom
    # return average

p_vals = {}
for narma_size in [5, 10, 20, 30]:
    for training_length in [500, 3000, 5000]:
        p_vals[(narma_size, training_length)] = experiment(narma_size, training_length)
        print(f"{narma_size=}, {training_length=}")

f = open("results.txt", "a")
f.write("\n")
f.write("brunner-munzel")
f.write(str(p_vals))
f.close()






