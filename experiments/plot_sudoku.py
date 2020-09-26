from matplotlib import pyplot as plt


def display_sudoku(tmplt, world, show_cands=False):
    tmplt.candidate_sets = {x: set(world.nodes[tmplt.is_cand[idx,:]]) for idx, x in enumerate(tmplt.nodes)}
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

def display_sudoku2(tmplt, world, show_cands=False):
    tmplt.candidate_sets = {x: set(world.nodes[tmplt.is_cand[idx,:]]) for idx, x in enumerate(tmplt.nodes)}
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
    plt.show()
