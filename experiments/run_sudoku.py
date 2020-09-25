import uclasm

from timeit import default_timer
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import csv

def display_sudoku(tmplt, show_cands=False):
    if not show_cands:
        # Easier to visualize result board
        print("-"*13)
        for i in range(9):
            row = "|"
            for j in range(9):
                square = chr(65+j)+str(i+1)
                if stype == "9x9x3":
                    square += "R"
                cands = tmplt.candidate_sets[square]
                digit = -1
                for cand in cands:
                    if digit == -1:
                        digit = cand[0]
                    elif cand[0] != digit:
                        digit = "X"
                row += str(digit)
                if j%3 == 2:
                    row += "|"
            print(row)
            if i%3 == 2:
                print("-"*13)
    else:
        # Candidate format
        print("-"*37)
        for i in range(9):
            rows = ["|", "|", "|"]
            for j in range(9):
                square = chr(65+j)+str(i+1)
                if stype == "9x9x3":
                    square += "R"
                cands = tmplt.candidate_sets[square]
                digit = -1
                possible = []
                for cand in cands:
                    if digit == -1:
                        digit = int(cand[0])
                        possible += [digit]
                    elif int(cand[0]) != digit:
                        if int(cand[0]) not in possible:
                            possible += [int(cand[0])]
                        digit = "X"
                # Print possibilities in a nice grid format
                for k in range(9):
                    if k+1 not in possible:
                        rows[k//3] += " "
                    else:
                        rows[k//3] += str(k+1)
                for m in range(3):
                    rows[m] += "|"
            for row in rows:
                print(row)
            print("-"*37)

def display_sudoku2(tmplt, show_cands=False):
    fig, ax = plt.subplots(figsize=(9,9))
    cur_axes = plt.gca()
    fig.patch.set_visible(False)
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)
    ax.axis("off")
    for i in range(10):
        plt.plot([0,9],[i,i],'k',linewidth=(5 if i%3 == 0 else 2))
        plt.plot([i,i],[0,9],'k',linewidth=(5 if i%3 == 0 else 2))
        if i == 9:
            continue
        for j in range(9):
            square = chr(65+j)+str(i+1)
            if stype == "9x9x3":
                square += "R"
            cands = tmplt.candidate_sets[square]
            digit = -1
            possible = []
            for cand in cands:
                if digit == -1:
                    digit = int(cand[0])
                    possible += [digit]
                elif int(cand[0]) != digit:
                    if int(cand[0]) not in possible:
                        possible += [int(cand[0])]
                    digit = "X"
            if len(possible) == 1:
                # Plot a large number
                plt.text(i+0.5, j+0.46, str(possible[0]), fontsize=32, ha='center', va='center')
            elif len(possible) > 0 and show_cands:
                for i2 in range(3):
                    for j2 in range(3):
                        digit = i2+j2*3+1
                        if digit in possible:
                            plt.text((i+i2/3.0)+1/6.0, (j+1-j2/3.0)-0.04-1/6.0, str(digit), fontsize=12, ha='center', va='center',weight='bold')

    plt.savefig("sudoku_picture{}.png".format("_cands" if show_cands else ""))


for stype in ['9x9','9x9x3','9x81']:
# stype = "9x9" # 9x9, 9x81, 9x9x3

    if stype == "9x9":
        channels = range(1)
    elif stype == "9x81":
        channels = range(3)
    elif stype == "9x9x3":
        channels = range(2)

    start_time = default_timer()

    size = 9 # Note: doesn't work with sizes other than 9x9, square link logic would have to be generalized as well as node labels

    if stype == "9x9":
        tmplt_adj_mats = [np.zeros((size*size,size*size), dtype=np.int8)]

        world_nodes = [str(i*10 + j) for i in range(1,size+1) for j in range(1,size+1)] # First digit actual digit, second digit is square it is in

        world_adj_mats = [np.ones((size*size,size*size), dtype=np.int8)] # Initialize to fully linked
        # Remove links between same digit
        for i in range(size):
            world_adj_mats[0][i*size:i*size+size, i*size:i*size+size] = 0

        tmplt_nodes = [chr(i)+str(j) for i in range(65,65+size) for j in range(1,size+1)] # Chessboard layout: letters are rows, numbers columns
        # Add links between rows
        link_mat = np.ones((size,size), dtype=np.int8) - np.eye((size), dtype=np.int8)
        for i in range(size):
            tmplt_adj_mats[0][i*size:i*size+size, i*size:i*size+size] = link_mat
        # Add links between columns
        for i in range(size):
            tmplt_adj_mats[0][i:i+size*(size-1)+1:size, i:i+size*(size-1)+1:size] = link_mat
        # Add links between same square
        for i in range(3):
            for j in range(3):
                row_idxs = [i*3+j*27+x for x in [0,1,2,9,10,11,18,19,20]] # i*3+j*27 = coordinate of top left corner of square
                tmplt_adj_mats[0][np.ix_(row_idxs, row_idxs)] = link_mat

        world_adj_mats[0] = sp.sparse.csr_matrix(world_adj_mats[0])
        tmplt_adj_mats[0] = sp.sparse.csr_matrix(tmplt_adj_mats[0])

    elif stype == "9x81":
        # 3 channels: row links, column links, square links

        tmplt_adj_mats = [np.zeros((size*size,size*size), dtype=np.int8) for i in range(3)]

        # Nodes in world graph: one node per digit per space
        # 3 character name: first character actual digit, 2-3rd are chessboard coordinate of space
        world_nodes = [str(k)+chr(i)+str(j) for k in range(1,size+1) for i in range(65,65+size) for j in range(1,size+1)]

        world_adj_mats = [np.zeros((len(world_nodes),len(world_nodes)), dtype=np.int8) for i in range(3)]
        # Add links between rows
        link_mat = np.ones((size*size,size*size), dtype=np.int8)
        for i in range(size):
            link_mat[i*size:i*size+size,i*size:i*size+size] = 0 # Remove same digit links
        for i in range(size):
            link_mat[i::size, i::size] = 0 # Remove same space links

        # Add links between rows
        for i in range(size):
            row_idxs = [i*size+j+k*size*size for j in range(size) for k in range(size)]
            world_adj_mats[0][np.ix_(row_idxs, row_idxs)] = link_mat
        # Add links between columns
        for i in range(size):
            world_adj_mats[1][i::size, i::size] = link_mat
        # Add links between same square
        for i in range(3):
            for j in range(3):
                square_idxs = [i*3+j*27+x+y*size*size for x in [0,1,2,9,10,11,18,19,20] for y in range(size)] # i*3+j*27 = coordinate of top left corner of square
                world_adj_mats[2][np.ix_(square_idxs, square_idxs)] = link_mat


        tmplt_nodes = [chr(i)+str(j) for i in range(65,65+size) for j in range(1,size+1)] # Chessboard layout: letters are rows, numbers columns
        # Add links between rows
        link_mat = np.ones((size,size), dtype=np.int8) - np.eye((size), dtype=np.int8)
        for i in range(size):
            tmplt_adj_mats[0][i*size:i*size+size, i*size:i*size+size] = link_mat
        # Add links between columns
        for i in range(size):
            tmplt_adj_mats[1][i:i+size*(size-1)+1:size, i:i+size*(size-1)+1:size] = link_mat
        # Add links between same square
        for i in range(3):
            for j in range(3):
                square_idxs = [i*3+j*27+x for x in [0,1,2,9,10,11,18,19,20]] # i*3+j*27 = coordinate of top left corner of square
                tmplt_adj_mats[2][np.ix_(square_idxs, square_idxs)] = link_mat

        for i in range(3):
            world_adj_mats[i] = sp.sparse.csr_matrix(world_adj_mats[i])
            tmplt_adj_mats[i] = sp.sparse.csr_matrix(tmplt_adj_mats[i])
    elif stype == "9x9x3":
        # 2 channels: adjacency links and same space links
        # Each type of link(row, column, square) has its own 9x9
        # Template is 3 copies of the squares, linked by same-space
        # First is row links, then col links, then square links

        tmplt_adj_mats = [np.zeros((size*size*3,size*size*3), dtype=np.int8) for i in range(2)]
        squares = [chr(i)+str(j) for i in range(65,65+size) for j in range(1,size+1)] # Chessboard layout: letters are rows, numbers columns
        tmplt_nodes = [x+y for y in ["R","C","S"] for x in squares]

        # Add links between rows
        link_mat = np.ones((size,size), dtype=np.int8) - np.eye((size), dtype=np.int8)
        for i in range(size):
            tmplt_adj_mats[0][i*size:i*size+size, i*size:i*size+size] = link_mat
        # Add links between columns
        for i in range(size):
            tmplt_adj_mats[0][size*size+i:size*size+i+size*(size-1)+1:size, size*size+i:size*size+i+size*(size-1)+1:size] = link_mat
        # Add links between same square
        for i in range(3):
            for j in range(3):
                row_idxs = [2*size*size+i*3+j*27+x for x in [0,1,2,9,10,11,18,19,20]] # i*3+j*27 = coordinate of top left corner of square
                tmplt_adj_mats[0][np.ix_(row_idxs, row_idxs)] = link_mat
        # Add same space links
        # Link from row to square and column to square
        link_mat2 = np.zeros((3,3))
        link_mat2[0,2] = 1
        link_mat2[1,2] = 1
        for i in range(size*size):
            tmplt_adj_mats[1][i::size*size, i::size*size] = link_mat2

        # Nodes in world graph: 3 nodes per digit per row/column/square
        # 3 character name: First digit actual digit, second digit is row/column/square it is in, third is R/C/S
        digits = [i*10 + j for i in range(1,size+1) for j in range(1,size+1)]
        world_nodes = [str(x)+y for y in ["R","C","S"] for x in digits]

        world_adj_mats = [np.zeros((len(world_nodes),len(world_nodes)), dtype=np.int8) for i in range(2)]
        # Initialize full links in row, column, square
        for i in range(3):
            world_adj_mats[0][i*size*size:(i+1)*size*size,i*size*size:(i+1)*size*size] = np.ones((size*size, size*size))
            # Remove links between same digit
            for j in range(size):
                world_adj_mats[0][i*size*size+j*size:i*size*size+j*size+size, i*size*size+j*size:i*size*size+j*size+size] = 0

        # Initialize same space links
        # Add links between same digit, from row to square and column to square
        # Only add a link if the row-square or column-square combo is legal
        link_row_square = np.zeros((2*size, 2*size))
        link_col_square = np.zeros((2*size, 2*size))
        for i in range(3):
            for j in range(3):
                link_row_square[i*3:i*3+3, size+i*3+j] = 1
                link_col_square[j*3:j*3+3, size+i*3+j] = 1

        for i in range(size):
            digit_idxs = [i*size+j for j in range(size)]
            rs_idxs = digit_idxs + [x+2*size*size for x in digit_idxs]
            cs_idxs = [x+size*size for x in digit_idxs] + [x+2*size*size for x in digit_idxs]
            world_adj_mats[1][np.ix_(rs_idxs, rs_idxs)] = link_row_square
            world_adj_mats[1][np.ix_(cs_idxs, cs_idxs)] = link_col_square

        for i in range(2):
            world_adj_mats[i] = sp.sparse.csr_matrix(world_adj_mats[i])
            tmplt_adj_mats[i] = sp.sparse.csr_matrix(tmplt_adj_mats[i])

    # initial candidate set for template nodes is the full set of world nodes
    tmplt = uclasm.Graph(np.array(tmplt_nodes), channels, tmplt_adj_mats)
    world = uclasm.Graph(np.array(world_nodes), channels, world_adj_mats)

    tmplt.is_cand = np.ones((tmplt.n_nodes,world.n_nodes), dtype=np.bool)
    tmplt.candidate_sets = {x: set(world.nodes) for x in tmplt.nodes}

    def update_node_candidates(tmplt, world, tmplt_node, cands):
        cand_row = np.zeros(world.n_nodes, dtype=np.bool)
        for cand in cands:
            cand_row[world.node_idxs[cand]] = True
        tmplt.is_cand[tmplt.node_idxs[tmplt_node]] &= cand_row

    tmplt.labels = np.array(['__'] * tmplt.n_nodes)
    world.labels = np.array(['__'] * world.n_nodes)

    if stype == "9x9":
        # Second digit is square
        # Restrict candidates to only allow a particular end digit in the square
        for i in range(3):
            for j in range(3):
                row_idxs = [i*3+j*27+x for x in [0,1,2,9,10,11,18,19,20]] # i*3+j*27 = coordinate of top left corner of square
                # for idx in row_idxs:
                #     cands = tmplt.candidate_sets[tmplt_nodes[idx]]
                #     new_cands = {cand for cand in cands if cand[-1] == str(i*3+j+1)}
                #     update_node_candidates(tmplt, world, tmplt_nodes[idx], new_cands)
                label = str(i) + str(j)
                tmplt.labels[row_idxs] = [label]*len(row_idxs)
                cand_idxs = [idx for idx, cand in enumerate(world.nodes) if cand[-1] == str(i*3+j+1)]
                world.labels[cand_idxs] = [label]*len(cand_idxs)
    elif stype == "9x81":
        # Restrict candidates to match the spaces
        for idx, tmplt_node in enumerate(tmplt_nodes):
            # cands = tmplt.candidate_sets[tmplt_node]
            # new_cands = {cand for cand in cands if cand[1:] == tmplt_node}
            # update_node_candidates(tmplt, world, tmplt_node, new_cands)
            label = str(tmplt_node)
            tmplt.labels[idx] = label
            new_cands = [idx for idx, cand in enumerate(world_nodes) if cand[1:] == tmplt_node]
            world.labels[new_cands] = label
    elif stype == "9x9x3":
        # Restrict candidates to match R/C/S
        # for i in range(size):
        #     for j in range(size):
        #         space = chr(65+i)+str(j+1)
        #         row_cands = set([str(k)+str(i+1)+"R" for k in range(1,size+1)])
        #         update_node_candidates(tmplt, world, space+"R", row_cands)
        #         col_cands = set([str(k)+str(j+1)+"C" for k in range(1,size+1)])
        #         update_node_candidates(tmplt, world, space+"C", col_cands)
        #         square = i//3 * 3 + j//3 + 1
        #         square_cands = set([str(k)+str(square)+"S" for k in range(1,size+1)])
        #         update_node_candidates(tmplt, world, space+"S", square_cands)
        for i in range(size):
            # Generate row labels
            row_label = str(i+1) + "R"
            row_tmplt = [chr(65+i)+str(j+1)+"R" for j in range(size)]
            row_cands = [str(k)+str(i+1)+"R" for k in range(1,size+1)]
            tmplt.labels[[tmplt.node_idxs[node] for node in row_tmplt]] = row_label
            world.labels[[world.node_idxs[node] for node in row_cands]] = row_label
            # Generate column labels
            col_label = str(i+1) + "C"
            col_tmplt = [chr(65+j)+str(i+1)+"C" for j in range(size)]
            col_cands = [str(k)+str(i+1)+"C" for k in range(1,size+1)]
            tmplt.labels[[tmplt.node_idxs[node] for node in col_tmplt]] = col_label
            world.labels[[world.node_idxs[node] for node in col_cands]] = col_label
            # Generate square labels
            square_label = str(i+1) + "S"
            square_cands = [str(k)+str(i+1)+"S" for k in range(1,size+1)]
            square_tmplt = [chr(65+x+(i//3)*3)+str(y+1+(i%3)*3)+"S" for x in range(3) for y in range(3)]
            tmplt.labels[[tmplt.node_idxs[node] for node in square_tmplt]] = square_label
            world.labels[[world.node_idxs[node] for node in square_cands]] = square_label

    tmplt_orig = tmplt
    world_orig = world

    for dataset in ['easy50','top95','hardest']:
        # Read in puzzles from Project Euler
        total_start_time = default_timer()
        filter_times = []
        validation_times = []
        iso_count_times = []
        iso_counts = []
        with open("sudoku-{}.txt".format(dataset), encoding="utf-8") as fin:
            # Format: 81 numbers, separated by newline
            for puzzle in fin:
                changed_nodes = np.zeros(tmplt.n_nodes, dtype=np.bool)
                start_time = default_timer()
                tmplt = tmplt_orig.copy()
                tmplt.is_cand = tmplt_orig.is_cand.copy()
                world = world_orig.copy()
                puzzle = puzzle.replace('\ufeff', '')
                puzzle = puzzle.replace('\n', '')

                for idx, char in enumerate(puzzle):
                    row = idx // 9 + 1 # One indexed
                    idx2 = idx % 9
                    if char in [str(x) for x in range(1,10)]: # Check nonzero
                        digit = int(char)
                        letter = chr(65+idx2)
                        if stype == "9x9":
                            update_node_candidates(tmplt, world,letter+str(row), set(world_nodes[(digit-1)*size:(digit-1)*size+size]))
                            changed_nodes[tmplt.node_idxs[letter+str(row)]] = True
                        elif stype == "9x81":
                            update_node_candidates(tmplt, world,letter+str(row), set([char+letter+str(row)]))
                            changed_nodes[tmplt.node_idxs[letter+str(row)]] = True
                        elif stype == "9x9x3":
                            for k in range(3):
                                link_types = ["R","C","S"]
                                update_node_candidates(tmplt, world,letter+str(row)+link_types[k], set(world_nodes[k*size*size+(digit-1)*size:k*size*size+(digit-1)*size+size]))
                                changed_nodes[tmplt.node_idxs[letter+str(row)+link_types[k]]] = True


        # # Read in a Sudoku puzzle to solve
        # with open("sudoku_puzzle2.txt") as fin:
        #     changed_nodes = np.zeros(tmplt.n_nodes, dtype=np.bool)
        #     row = 1
        #     for line in fin: # Each line has 9 characters. Characters not in {1,9} are considered blanks
        #         for idx2, char in enumerate(line):
        #             if char in [str(x) for x in range(1,10)]:
        #                 digit = int(char)
        #                 letter = chr(65+idx2)
        #                 if stype == "9x9":
        #                     update_node_candidates(tmplt, world,letter+str(row), set(world_nodes[(digit-1)*size:(digit-1)*size+size]))
        #                     changed_nodes[tmplt.node_idxs[letter+str(row)]] = True
        #                 elif stype == "9x81":
        #                     update_node_candidates(tmplt, world,letter+str(row), set([char+letter+str(row)]))
        #                     changed_nodes[tmplt.node_idxs[letter+str(row)]] = True
        #                 elif stype == "9x9x3":
        #                     for k in range(3):
        #                         link_types = ["R","C","S"]
        #                         update_node_candidates(tmplt, world,letter+str(row)+link_types[k], set(world_nodes[k*size*size+(digit-1)*size:k*size*size+(digit-1)*size+size]))
        #                         changed_nodes[tmplt.node_idxs[letter+str(row)+link_types[k]]] = True
        #         row += 1

        #        tmplt.summarize_candidate_sets()
                # print("Time to create world and template: {}".format(default_timer()-start_time))

                # tmplt.candidate_sets = {x: set(world.nodes[tmplt.is_cand[idx,:]]) for idx, x in enumerate(tmplt.nodes)}
                # display_sudoku2(tmplt, show_cands=False)

                start_time = default_timer()
                tmplt, world, candidates = uclasm.run_filters(tmplt, world, candidates=tmplt.is_cand,
                    init_changed_cands=changed_nodes, filters=uclasm.all_filters, verbose=False)
                print("Time taken for filters: {}".format(default_timer()-start_time))
                filter_times += [default_timer()-start_time]
                start_time = default_timer()
                from filters.validation_filter import validation_filter
                validation_filter(tmplt, world, candidates=candidates, in_signal_only=False,
                      verbose=False)
                print("Time taken for validation: {}".format(default_timer()-start_time))
                validation_times += [default_timer()-start_time]
                # # tmplt.candidate_sets = {x: set(world.nodes[candidates[idx,:]]) for idx, x in enumerate(tmplt.nodes)}

                # print("Starting isomorphism count")
                start_time = default_timer()
                count = uclasm.count_isomorphisms(tmplt, world, candidates=candidates, verbose=False)
                print("Counted {} isomorphisms in {} seconds".format(count, default_timer()-start_time))
                count = 1
                iso_counts += [count]
                iso_count_times += [default_timer()-start_time]
        print("Dataset:", dataset)
        print("Representation:", stype)
        print("Total time for {} puzzles: {}".format(len(filter_times),default_timer()-total_start_time))
        print("Time spent filtering: {}".format(sum(filter_times)))
        print("Time spent counting isomorphisms: {}".format(sum(iso_count_times)))

        total_times = np.array(filter_times)+np.array(iso_count_times)

        np.save('sudoku_times_{}_{}_validation.npy'.format(dataset,stype), total_times)
        np.save('sudoku_filter_times_{}_{}_validation.npy'.format(dataset,stype), filter_times)
        np.save('sudoku_validation_times_{}_{}_validation.npy'.format(dataset,stype), validation_times)
        np.save('sudoku_iso_count_times_{}_{}_validation.npy'.format(dataset,stype), iso_count_times)
        np.save('sudoku_iso_counts_{}_{}_validation.npy'.format(dataset,stype), iso_counts)
