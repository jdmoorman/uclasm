"""Provides a method for finding the best k matchings using A* search"""
from uclasm import MatchingProblem

from heapq import heappush, heappop, nlargest

class State():
    """A node class for A* pathfinding"""
    def __init__(self):
        self.state = None
        self.parent = None
        self.child = None
        self.n_determined = 0
        self.f = 0
        self.g = float("inf")

    def copy(self):
        r = State()
        r.state = self.state.copy()
        r.parent = self.parent
        r.n_determined = self.n_determined
        r.f = self.f
        r.g = self.g

    def is_end(self):
        return n_determined == self.state.tmplt.n_nodes

    def __lt__(self, other):
        if self.n_determined > other.n_determined:
            return True
        elif self.n_determined == other.n_determined:
            return self.f < other.f
        else:
            return False

def a_star_best_matching(smp, k=1):
    """Performs A* search using the local cost heuristic"""
    start_state = State()
    start_state.state = smp
    start_state.g = 0

    solution = []

    open_list = []
    closed_list = []

    while len(open_list) > 0:
        current_state = open_list[0]

        if current_state.is_end():
            solution.append(current_state)
            if len(solution) >= k:
                return solution
            heappop(open_list)
            continue

        heappush(closed_list, current_state)
        heappop(open_list)

        for i in unspec_nodes:
            child = create_child(assigning unspec node i)
            tentative_g = current_state.g + dist(current_state, child)
            if tentative_g < :
                child.parent = current_state
                child.
            else:
                del child
