import uclasm

from matplotlib import pyplot as plt
import matplotlib.colors as colors
import numpy as np
import scipy as sp
import pandas as pd

n_layers = 1
# n_trials = 10
# n_trials = 500
n_trials = 40
# n_layers = 1
count_isomorphisms = False
timeout = 1000

if False:
    results = np.load("erdos_renyi_results_{}_trials_{}_layers.npy".format(n_trials, n_layers), allow_pickle=True)
    # n_tmplt_nodes = 10
    # results = np.load("erdos_renyi_results_{}_trials_{}_layers_{}_tnodes_vary_probs.npy".format(n_trials, n_layers, n_tmplt_nodes), allow_pickle=True)

    results_df = pd.DataFrame([x for x in results])

    # plot 1: number of search nodes vs. template and world probability

    plt.figure(1)

    mean_iters = np.zeros((9, 9))
    tmplt_probs = [x/10.0 for x in range(1,10)]
    world_probs = [x/10.0 for x in range(1,10)]
    for tmplt_prob_idx, tmplt_prob in enumerate(tmplt_probs):
        for world_prob_idx, world_prob in enumerate(world_probs):
            matching_results = results_df.query('tmplt_prob == {} and world_prob == {}'.format(tmplt_prob, world_prob))
            if len(matching_results.index) != n_trials: # Some trials lost to timeout
                pass
            else:
                # These indices are ordered (y, x) by plt.pcolor, don't ask me why...
                mean_iters[world_prob_idx, tmplt_prob_idx] = np.mean(matching_results['n_iterations'])

    mean_iters[mean_iters == 0.0] = 10 ** np.ceil(np.log10(np.max(mean_iters)))

    # plt.imshow(mean_iters, origin='lower')
    plt.pcolor(world_probs, tmplt_probs, mean_iters, norm=colors.LogNorm(vmin=np.min(mean_iters), vmax=np.max(mean_iters)))
    plt.colorbar()

    plt.savefig("n_iter_vs_tmplt_world_prob.pdf")

# plot 2: number of search nodes vs. number of world nodes, holding template nodes fixed

# for n_layers in [1,3,5,7,9]:
if True:
    plt.figure()
    # mean_iters = []

    # for n_world_nodes in range(10,300,10):
    #   results = np.load("erdos_renyi_results_{}_trials_{}_layers_{}_wnodes_vary_world_size.npy".format(n_trials, n_layers, n_world_nodes), allow_pickle=True)
    #   results_df = pd.DataFrame([x for x in results])
    #   if len(results_df.index):
    #       mean_iters.append(np.mean(results_df['n_iterations']))
    #   else:
    #       mean_iters.append(0)
    # plt.plot(range(10,300,10), mean_iters)

    results = np.load("erdos_renyi_results_{}_trials_{}_layers{}_timeout_{}_vary_world_size.npy".format(n_trials, n_layers, "_count_iso" if count_isomorphisms else "", timeout), allow_pickle=True)
    results_df = pd.DataFrame([x for x in results])
    w_nodes_list = pd.unique(results_df['n_world_nodes'])
    mean_iters = []
    lower_ci = []
    upper_ci = []
    ci = 95
    median_iters = []
    mean_times = []
    median_times = []
    max_times = []
    n_trials_success = []
    for n_world_nodes in w_nodes_list:
        n_iterations = results_df[results_df.n_world_nodes == n_world_nodes]['n_iterations']
        mean_iters.append(n_iterations.mean())
        lower_ci.append(np.percentile(n_iterations, 50-ci/2, axis=0))
        upper_ci.append(np.percentile(n_iterations, 50+ci/2, axis=0))
        median_iters.append(np.percentile(n_iterations, 50, axis=0))
        times = results_df[results_df.n_world_nodes == n_world_nodes]['iso_count_time' if count_isomorphisms else 'iso_check_time']
        mean_times.append(times.mean())
        median_times.append(np.percentile(times, 50, axis=0))
        max_times.append(times.max())
        n_trials_success.append(len(n_iterations))
    print(n_trials_success)
    plt.plot(w_nodes_list, mean_iters)
    plt.plot(w_nodes_list, np.polyval(np.polyfit(w_nodes_list, mean_iters, 2), w_nodes_list), 'k')
    plt.xlabel("Number of World Nodes")
    plt.ylabel("Mean Number of Iterations")
    if count_isomorphisms:
        plt.title("Mean Number of Iterations for Isomorphism Counting")
    else:
        plt.title("Mean Number of Iterations for Isomorphism Checking")

    plt.savefig("n_iter_vs_n_world_nodes_{}_layers_{}_trials{}.pdf".format(n_layers, n_trials, "_count_iso" if count_isomorphisms else ""))
    plt.fill_between(w_nodes_list, lower_ci, upper_ci,
                     alpha=0.25)
    plt.savefig("n_iter_vs_n_world_nodes_{}_layers_{}_trials_{}_ci{}.pdf".format(n_layers, n_trials, ci, "_count_iso" if count_isomorphisms else ""))
    print(w_nodes_list, mean_iters)

    plt.figure()
    plt.plot(w_nodes_list, median_iters)
    plt.xlabel("Number of World Nodes")
    plt.ylabel("Median Number of Iterations")
    plt.title("Median Number of Iterations for Isomorphism "+ ("Counting" if count_isomorphisms else "Checking"))
    plt.savefig("n_iter_vs_n_world_nodes_{}_layers_{}_trials{}_median.pdf".format(n_layers, n_trials, "_count_iso" if count_isomorphisms else ""))
    plt.fill_between(w_nodes_list, lower_ci, upper_ci,
                     alpha=0.25)
    plt.savefig("n_iter_vs_n_world_nodes_{}_layers_{}_trials_median_{}_ci{}.pdf".format(n_layers, n_trials, ci, "_count_iso" if count_isomorphisms else ""))

    plt.figure()
    plt.plot(w_nodes_list, mean_times)
    deg = np.polyfit(w_nodes_list, mean_times, 2)
    print(deg)
    plt.plot(w_nodes_list, np.polyval(deg, w_nodes_list), 'k')
    plt.xlabel("Number of World Nodes")
    plt.ylabel("Mean Time Per Trial")
    plt.title("Mean Time Per Trial for Isomorphism "+ ("Counting" if count_isomorphisms else "Checking"))
    plt.savefig("time_vs_n_world_nodes_{}_layers_{}_trials{}.pdf".format(n_layers, n_trials, "_count_iso" if count_isomorphisms else ""))

    plt.figure()
    plt.plot(w_nodes_list, median_times)
    plt.xlabel("Number of World Nodes")
    plt.ylabel("Median Time Per Trial")
    plt.title("Median Time Per Trial for Isomorphism "+ ("Counting" if count_isomorphisms else "Checking"))
    plt.savefig("time_vs_n_world_nodes_{}_layers_{}_trials{}_median.pdf".format(n_layers, n_trials, "_count_iso" if count_isomorphisms else ""))

    plt.figure()
    plt.plot(w_nodes_list, max_times)
    deg = np.polyfit(w_nodes_list, max_times, 2)
    print(deg)
    plt.plot(w_nodes_list, np.polyval(deg, w_nodes_list), 'k')
    plt.xlabel("Number of World Nodes")
    plt.ylabel("Max Time Per Trial")
    plt.title("Max Time Per Trial for Isomorphism "+ ("Counting" if count_isomorphisms else "Checking"))
    plt.savefig("time_vs_n_world_nodes_{}_layers_{}_trials{}_max.pdf".format(n_layers, n_trials, "_count_iso" if count_isomorphisms else ""))


# plot 3: number of search nodes vs. number of layers
#  option 1: hold total number of edges fixed
#  option 2: number of edges proportional to number of layers
