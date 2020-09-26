from matplotlib import pyplot as plt
plt.switch_backend('agg')
import numpy as np

datasets = ["easy50", "top95", "hardest"]
methods = ["9x9", "9x9x3"]
colors = ['r','b']
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams.update({'font.size': 14})

#fig1, f1_axes = plt.subplots(ncols=2, nrows=3, constrained_layout=True)

plt.figure()

for i, dataset in enumerate(datasets):
    for j, method in enumerate(methods):
        times = np.load("sudoku_times_{}_{}_validation.npy".format(dataset, method))
        plt.subplot(3, 2, i*2+j+1)
        plt.hist(times, color=colors[j], label="{}_{}".format(method, dataset))
        plt.legend()

plt.savefig("test_sudoku_hist.png")

plt.figure("all_scatter", figsize=(6,3))

markers = ["*", "o", "D"]
all_min = float("inf")
all_max = 0

for i, dataset in enumerate(datasets):
    times1 = np.load("sudoku_times_{}_{}_validation.npy".format(dataset, methods[0]))
    times2 = np.load("sudoku_times_{}_{}_validation.npy".format(dataset, methods[1]))
    plt.figure()
    plt.scatter(times1, times2, marker=markers[i])
    plt.xlabel(methods[0]+" time (s)", fontsize=14)
    plt.ylabel(methods[1]+" time (s)", fontsize=14)
    plt.yscale("log")
    plt.xscale("log")
    min_val = min(np.min(times1), np.min(times2))
    max_val = max(np.max(times1), np.max(times2))
    plt.plot([min_val, max_val], [min_val, max_val], color='k')
    plt.savefig("test_sudoku_scatter_{}_log.pdf".format(dataset))
    all_min = min(all_min, min_val)
    all_max = max(all_max, max_val)


for i, dataset in enumerate(datasets):
    times1 = np.load("sudoku_times_{}_{}_validation.npy".format(dataset, methods[0]))
    times2 = np.load("sudoku_times_{}_{}_validation.npy".format(dataset, methods[1]))
    plt.figure("all_scatter")
    plt.scatter(times1, times2, marker=markers[i], label=dataset, zorder=1)
    plt.xlabel(methods[0]+" time (s)", fontsize=14)
    plt.ylabel(methods[1]+" time (s)", fontsize=14)
    plt.yscale("log")
    plt.xscale("log")
    #plt.plot([min_val, max_val], [min_val, max_val], color='k')
plt.figure("all_scatter")
plt.plot([all_min, all_max], [all_min, all_max], color='k', zorder=0, label="y=x")
plt.legend()
plt.tight_layout(pad=0.2)
plt.savefig("test_sudoku_scatter_all_log.pdf")
