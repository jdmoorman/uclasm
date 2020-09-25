import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
import sys
sys.path.append('../')
import uclasm

from uclasm.filters.validation_filter import validation_filter

def plot_barcharts(tmplt, world, results_dir, data_name,
                   neighborhood=True, elimination=True, validation=False):
    """
    Plot (1) number of candidates of each template node (2) number of template
    nodes that each world node is the candidate of
    """
    filters = [uclasm.stats_filter]
    tmplt, world, candidates = uclasm.run_filters(tmplt, world,
                                                  filters=filters,
                                                  verbose=False,
                                                  reduce_world=True)
    cands_stats = candidates.sum(axis=1)
    cinvs_stats = candidates.sum(axis=0)

    filters.append(uclasm.topology_filter)
    tmplt, world, candidates = uclasm.run_filters(tmplt, world,
                                                  candidates=candidates,
                                                  filters=filters,
                                                  verbose=False,
                                                  reduce_world=True)
    cands_topo = candidates.sum(axis=1)
    cinvs_topo = candidates.sum(axis=0)

    if neighborhood:
        filters.append(uclasm.neighborhood_filter)
        tmplt, world, candidates = uclasm.run_filters(tmplt, world,
                                                      candidates=candidates,
                                                      filters=filters, verbose=False,
                                                      reduce_world=False)
        cands_nbhd = candidates.sum(axis=1)
        cinvs_nbhd = candidates.sum(axis=0)

    if elimination:
        tmplt, world, candidates = uclasm.run_filters(tmplt, world,
                                                      candidates=candidates,
                                                      filters=uclasm.all_filters,
                                                      verbose=False,
                                                      reduce_world=False)
        cands_elim = candidates.sum(axis=1)
        cinvs_elim = candidates.sum(axis=0)

    if validation:
        tmplt, world, candidates = validation_filter(tmplt, world,
                                                     candidates=candidates,
                                                     verbose=False,
                                                     reduce_world=False)
        cands_valid = candidates.sum(axis=1)
        cinvs_valid = candidates.sum(axis=0)

    # Sort by # of candidates left after elim, topo, stats
    if validation:
        order_tmplt = sorted(range(len(tmplt.nodes)), \
            key=lambda idx: (cands_valid[idx], cands_elim[idx], cands_nbhd[idx],\
                            cands_topo[idx], cands_stats[idx]))

        order_world = sorted(range(len(world.nodes)), \
            key=lambda idx: (cinvs_valid[idx], cinvs_elim[idx], cinvs_nbhd[idx],\
                            cinvs_topo[idx], cinvs_stats[idx]))
    elif elimination:
        order_tmplt = sorted(range(len(tmplt.nodes)), \
            key=lambda idx: (cands_elim[idx], \
                            cands_topo[idx], cands_stats[idx]))

        order_world = sorted(range(len(world.nodes)), \
            key=lambda idx: (cinvs_elim[idx], \
                            cinvs_topo[idx], cinvs_stats[idx]))
    else:
        order_tmplt = sorted(range(len(tmplt.nodes)), \
            key=lambda idx: (cands_nbhd[idx], \
                            cands_topo[idx], cands_stats[idx]))

        order_world = sorted(range(len(world.nodes)), \
            key=lambda idx: (cinvs_nbhd[idx], \
                            cinvs_topo[idx], cinvs_stats[idx]))

    # Reorganize the candidates
    cands_stats = np.array([cands_stats[i] for i in order_tmplt])
    cands_topo = np.array([cands_topo[i] for i in order_tmplt])

    cinvs_stats = np.array([cinvs_stats[i] for i in order_world])
    cinvs_topo = np.array([cinvs_topo[i] for i in order_world])

    # Keep only the world nodes that are still candidates to at least one
    # tmplt node after topology filter
    world_to_keep = np.nonzero(cinvs_topo)[0]
    cinvs_stats = cinvs_stats[world_to_keep]
    cinvs_topo = cinvs_topo[world_to_keep]

    if neighborhood:
        cands_nbhd = np.array([cands_nbhd[i] for i in order_tmplt])
        cinvs_nbhd = np.array([cinvs_nbhd[i] for i in order_world])
        cinvs_nbhd = cinvs_nbhd[world_to_keep]

    if elimination:
        cands_elim = np.array([cands_elim[i] for i in order_tmplt])
        cinvs_elim = np.array([cinvs_elim[i] for i in order_world])
        cinvs_elim = cinvs_elim[world_to_keep]

    if validation:
        cands_valid = np.array([cands_valid[i] for i in order_tmplt])
        cinvs_valid = np.array([cinvs_valid[i] for i in order_world])
        cinvs_valid = cinvs_valid[world_to_keep]

    y1 = np.arange(len(tmplt.nodes))
    y2 = np.arange(len(world_to_keep))

    plt.rcParams.update({'font.size': 20})
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams['figure.figsize'] = (15, 12)

    _, color1, color2, color3, color4, color5 = sns.color_palette("Blues", n_colors=6)

    ax1 = plt.subplot(211)

    # Plot tmplt bars
    # plt.figure()
    plt.bar(y1, height=cands_stats, align='center', color=color1, \
                alpha=1, width=1, label='After Statistics')
    plt.bar(y1, height=cands_topo,  align='center', color=color2, \
                alpha=1, width=1, label='After Topology')
    if neighborhood:
        plt.bar(y1, height=cands_nbhd, align='center', color=color3, \
                    alpha=1, width=1, label='After Neighborhood')
    if elimination:
        plt.bar(y1, height=cands_elim, align='center', color=color4, \
                    alpha=1, width=1, label='After Elimination')
    if validation:
        plt.bar(y1, height=cands_valid, align='center', color=color5, \
                    alpha=1, width=1, label='After Validation')

    plt.yscale('log')
    plt.xlabel('Template Node No.', fontsize=20)
    plt.ylabel('Number of Candidates', fontsize=20)
    # plt.ylim(0, 2500)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=1, fancybox=True, shadow=True)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.savefig('{}/{}_tmplt_bars_new.png'.format(results_dir, data_name), bbox_inches='tight')
    # plt.close()

    ax2 = plt.subplot(212)

    # Plot world bars
    # plt.figure()
    plt.bar(y2, height=cinvs_stats, align='center', color=color1, \
                alpha=1, width=1, label='After Statistics')
    plt.bar(y2, height=cinvs_topo, align='center',color=color2, \
                alpha=1, width=1, label='After Topology')
    if neighborhood:
        plt.bar(y2, height=cinvs_nbhd, align='center', color=color3, \
                    alpha=1, width=1, label='After Neighborhood')
    if elimination:
        plt.bar(y2, height=cinvs_elim, align='center', color=color4, \
                    alpha=1, width=1, label='After Elimination')
    if validation:
        plt.bar(y2, height=cinvs_valid, align='center', color=color5, \
                    alpha=1, width=1,label='After Validation')
    plt.xlabel('World Node No.', fontsize=20)
    plt.ylabel('Number of Template Nodes', fontsize=20)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                                        fancybox=True, shadow=True, ncol=2)
    plt.ylim(0, 40)
    # plt.savefig('{}/{}_world_bars_new.png'.format(results_dir, data_name),bbox_inches='tight')
    plt.savefig('{}/{}_bars.png'.format(results_dir, data_name), bbox_inches='tight')
    plt.close()
