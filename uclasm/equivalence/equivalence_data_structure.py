"""
Implementation of an equivalence data structure for partitioning
vertices into equivalence classes. From Pattis' ICS46 lecture's note:
https://www.ics.uci.edu/~pattis/ICS-46/lectures/notes/equivalence.txt
by TimNg
"""

from collections import defaultdict
from copy import deepcopy

class Equivalence:
    """
    Equivalence class data structure for efficient equivalence classes
    computation. This is implemented using the disjoint sets data structure.
    See https://en.wikipedia.org/wiki/Disjoint-set_data_structure for a basic
    overview of the data structure.

    Internally, this class maintains a dictionary containing the mapping
    of every element to its parent (which may be itself) called parent_map.
    Searches for which class an element is in amounts to traversing through
    the parents until it finds an element which is its own parent, and then
    this is the representative element.
    """
    def __init__(self, starting_set = set()):
        """
        Starting_set is a set of node to start with.
        """
        # dictionary of parents where each key is mapped to its parent
        # start with everything maps to itself
        self.parent_map = {a: a for a in starting_set}
        # dictionary of root: size_of_tree
        # also serves as something to store the roots
        self.root_size_map = {a : 1 for a in starting_set}

    ### METHODS ###
    def add_singleton(self, new_value):
        """
        Add a new value as its own equiv class
        """
        # Check first that it is not already in a member
        error_msg = ("Equivalence.add_singleton: {} is already " + 
                    "in the equiv class!".format(new_value))
        assert new_value not in self.parent_map, error_msg

        self.parent_map[new_value] = new_value
        self.root_size_map[new_value] = 1

    def merge_classes_of(self, a , b):
        """
        Merge the equivalence classes of a and b together
        by setting each parent's map to point to the same root.
        """
        assert a in self.parent_map, "Value " + str(a) + " does not exist."
        assert b in self.parent_map, "Value " + str(b) + " does not exist."

        # get the roots while compressing the tree
        root_of_a = self.compress_to_root(a)
        root_of_b = self.compress_to_root(b)

        if (root_of_a == root_of_b):
            return # they are already in the same equivalence class

        # find the "big" root and the "small" root
        if self.root_size_map[root_of_a] < self.root_size_map[root_of_b]:
            small_root, big_root = root_of_a, root_of_b
        else:
            small_root, big_root = root_of_b, root_of_a

        # now we change the root of the smaller size map to the bigger one
        # then we update the size of the new equiv class and then remove 
        # the smaller one as a root
        self.parent_map[small_root] = big_root
        self.root_size_map[big_root] += self.root_size_map[small_root]

        # remove the small_root from the roots dict
        del self.root_size_map[small_root]  

    def merge_set(self, set_to_merge):
        """
        Given a set of elements in Equivalence, merge them together
        """
        for a in set_to_merge:
            assert a in self.parent_map, "Value " + str(a) + " does not exist."
        one_elem = next(iter(set_to_merge))
        for a in set_to_merge:
            self.merge_classes_of(one_elem, a)

    def partition(self, equivalence_relation: 'function', *args, **kwargs):
        """
        Given an equivalence_relation function compute the appropriate 
        equivalence classes of the elements in the Equivalence data structure. 
        Users have to make sure of semantic and care should be taken to make 
        sure the equivalence_relation passed is an actual equiv relation,
        otherwise the output will be undefined.
        --> Checking in_same_class(a,b) will be the same as using
            equivalence_relation(a,b)
        This uses a divide and conquer approach on the root_size_map dictionary.
        """
        classes = []
        for elem in self.root_size_map:
            found = False
            for c in classes:
                # if these two classes are equivalence then we add them
                if equivalence_relation(next(iter(c)), elem, *args, *kwargs):
                    c.add(i)
                    found = True
                    break
            if not found: # new class
                classes.append(set([elem]))
        for c in classes:
            self.merge_set(c)
        return classes

    ### QUERIES ###
    def in_same_class(self, a, b) -> bool:
        """
        return a bool indicating if a and b are in the same classes
        """
        assert a in self.parent_map, str(a) + " does not exist."
        assert b in self.parent_map, str(b) + " does not exist."
        root_of_a = self.compress_to_root(a)
        root_of_b = self.compress_to_root(b)
        return root_of_a == root_of_b

    def __len__(self):
        """
        Returns the number of elements in the equiv class
        """
        return len(self.parent_map)

    def class_count(self):
        """
        return the number of unique equiv classes
        """
        return len(self.root_size_map)

    @property
    def classes(self):
        """
        This function returns a dictionary whose values are the equivalence
        classes and whose keys are the representatives of the classes.
        """
        answer_map = defaultdict(set)
        for elem in self.parent_map:
            class_rep = self.representative(elem)
            answer_map[class_rep].add(elem)
        return answer_map

    def get_class(self, elem):
        return self.classes[self.representative(elem)]

    def non_trivial_classes(self):
        """
        This function returns a dictionary whose values are the equivalence
        classes and whose keys are the representatives of the classes.
        The dictionary will only contain classes which have more than one
        element.
        """
        return {rep: class_ for (rep, class_) in self.classes.items()
                if len(class_) > 1}

    def __str__(self):
        return "Equiv Classes: {}\nParent_map: {}\nRoot_size_map: {}".format(
                self.classes, self.parent_map, self.root_size_map)

    def __getitem__(self, key):
        """
        Return the equivalence class for the passed in key.
        """
        rep = self.representative(key)
        return self.classes[rep]

    def compress_to_root(self, elem) -> 'root of a':
        """
        Returns the root of elem, and on the way, set the roots of the 
        parents of elem to be the root of elem
        """
        assert elem in self.parent_map, str(elem) + " does not exist."
        parents_of_elem = set() # set to store the parents of elem
        curr_val = elem   # current value

        # We traverse up through the ancestors of elem and keep track
        # of the elements that we see so that we can compress the path
        while curr_val != self.parent_map[curr_val]:
            parents_of_elem.add(curr_val)
            curr_val = self.parent_map[curr_val]

        # curr_val is now the root
        # now we set all the parents of elem to the root
        for parent in parents_of_elem:
            self.parent_map[parent] = curr_val
        return curr_val

    def representative(self, elem):
        """
        Return the representative of elem's equivalence class.
        This performs path compression in the process of finding the 
        representative.
        """
        return self.compress_to_root(elem)

    def copy(self):
        """
        Return a deep copy of self
        """
        return deepcopy(self)


def equivalence_from_partition(partition):
    """
    Given a partition of some set, create an Equivalence object where two
    elements are equivalent if they are in the same cell in the partition.
    This isn't as efficient as it could be, but it suffices for now.
    """ 
    elems = [x for xs in partition for x in xs]

    equiv = Equivalence(elems)
    for xs in partition:
        equiv.merge_set(xs)

    return equiv
