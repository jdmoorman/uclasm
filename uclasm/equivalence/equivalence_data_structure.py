""" Implementation of an equivalence data structure for partitioning
	vertices into equivalence classes. From Pattis' ICS46 lecture's note:
	https://www.ics.uci.edu/~pattis/ICS-46/lectures/notes/equivalence.txt
	by TimNg
	Last update: 7/8/19 (basic implementation)"""

from collections import defaultdict


class Equivalence(object):
	""" Equivalence class data structure for efficient equivalence classes
	computation"""
	def __init__(self, starting_set):
		""" starting_set is a set of node to start with """
		# dictionary of parents where each key is mapped to its parent
		# start with everything maps to itself
		self.parent_map = {a: a for a in starting_set}
		# dictionary of root: size_of_tree
		# also serves as something to store the roots
		self.root_size_map = {a: 1 for a in starting_set}

	# ## METHODS ###
	def add_singleton(self, new_value):
		""" add a new value as its own equiv class"""
		assert new_value not in self.parent_map, "Equivalence.add_singleton: "+str(new_value) + " is already in the equiv class!"
		self.parent_map[new_value] = new_value
		self.root_size_map[new_value] = 1

	def merge_classes_of(self, a , b):
		""" merge the equivalence classes of a and b together
		by setting each parent's map to point to the same root """
		assert a in self.parent_map, "Equivalence.merge_classes_of: Value " + str(a) + " does not exist."
		assert b in self.parent_map, "Equivalence.merge_classes_of: Value " + str(b) + " does not exist."
		# get the roots while compressing the tree
		root_of_a = self.compress_to_root(a)
		root_of_b = self.compress_to_root(b)
		if root_of_a == root_of_b:
			return # they are already in the same equivalence class
		# find the "big" root and the "small" root
		small_root, big_root = (root_of_a, root_of_b) if self.root_size_map[root_of_a] < self.root_size_map[root_of_b] \
					 else (root_of_b, root_of_a)
		# now we change the root of the smaller size map to the bigger one
		# then we update the size of the new equiv class and then remove the smaller one
		# as a root
		self.parent_map[small_root] = big_root
		self.root_size_map[big_root] += self.root_size_map[small_root] # we grew!
		del self.root_size_map[small_root]  # remove the small_root from the roots dict

	def merge_set(self, set_to_merge):
		""" given a set of elements in Equivalence, merge them together """
		for a in set_to_merge:
			assert a in self.parent_map, "Equivalence.merge_set: Value " + str(a) + " does not exist."
		one_elem = next(iter(set_to_merge))
		for a in set_to_merge: # more efficient way is to merge pairwise???
			self.merge_classes_of(one_elem, a)

	def partition(self, equivalence_relation: 'function', *args, **kwargs) -> [set]:
		""" given an equivalence_relation function compute the appropriate equiv classes
		of the elements in the Equivalence data structure. Users have to make sure of semantic
		and care should be taken to make sure the equivalence_relation passed is an actual equiv relation, otherwise
		the output will be undefined.
		--> Checking in_same_class(a,b) will be the same as using equivalence_relation(a,b)
		This uses a divide and conquer approach on the root_size_map dictionary.
		The time complexity should be n*logn (need a proof though...)
			for a divide and conquer approach ... now it's just naive implementation """
		classes = []
		# TODO: parallelize
		for i in self.root_size_map:
			found = False
			for c in classes:
				# if these two classes are equivalence then we add them
				if equivalence_relation(next(iter(c)), i, *args, *kwargs):
					c.add(i)
					found = True
					break
			if not found:  # new class
				classes.append({i})
		for c in classes:
			self.merge_set(c)
		return classes

	# ## QUERIES ###
	def in_same_class(self, a, b) -> bool:
		""" return a bool indicating if a and b are in the same classes """
		assert a in self.parent_map, "Equivalence.in_same_class: " + str(a) + " does not exist."
		assert b in self.parent_map, "Equivalence.in_same_class: " + str(b) + " does not exist."
		root_of_a = self.compress_to_root(a)
		root_of_b = self.compress_to_root(b)
		return root_of_a == root_of_b

	def __len__(self):
		""" returns the number of elements in the equiv class"""
		return len(self.parent_map)

	def class_count(self):
		""" return the number of unique equiv classes"""
		return len(self.root_size_map)

	def classes(self) -> [{'equiv classes'}]:
		""" returns a list of sets of equivalence classes (python doesn't allow set of sets)"""
		answer_map = defaultdict(set) # lookup defaultdict if not sure. just instantiate an empty set by default
		for i in self.parent_map:
			root_of_i = self.compress_to_root(i)
			answer_map[root_of_i].add(i)
		return list(answer_map.values())

	def non_trivial_classes(self) -> [{int}]:
		""" returns a list of sets of non-trivial (>1) equivalence classes\
		(python doesn't allow set of sets)"""
		answer_map = defaultdict(set) # lookup defaultdict if not sure. just instantiate an empty set by default
		for i in self.parent_map:
			root_of_i = self.compress_to_root(i)
			answer_map[root_of_i].add(i)
		return [i for i in answer_map.values() if len(i)>1]

	def __str__(self):
		""" print out the equiv classes """
		return "Equiv Classes: " + str(self.classes()) + "\nParent_map: " + str(self.parent_map)\
			+ "\nRoot_size_map: " + str(self.root_size_map)

	def __repr__(self):
		result = f"#classes/#vertices={self.class_count()}/{len(self)}. " \
			f"Non-triv classes size: {[len(i) for i in self.non_trivial_classes()]}"
		return result

	def get_equiv_size(self, node):
		return self.root_size_map[self.compress_to_root(node)]

	def compress_to_root(self, a) -> 'root of a':
		""" returns the root of a, and on the way, set the roots of the parents of a
		to be the root of a"""
		assert a in self.parent_map, "Equivalence.compress_to_root: "+ str(a) + "does not exist."
		parents_of_a = set()  # set to store the parents of a
		curr_val = a   # current value
		while curr_val != self.parent_map[curr_val]:
			# while the current value is not its own parent (otherwise it's the root)
			parents_of_a.add(curr_val)  # we add it to the set of parents_of_a
			curr_val = self.parent_map[curr_val] # we keep going to the next parent
		# note that curr_val is now the root: the condition to break
		# now we set all the parents of a to the root, this is the compression step
		for i in parents_of_a:
			self.parent_map[i] = curr_val
		return curr_val

	def get_all_roots(self) -> {int or str}:
		""" Return all the roots of the equiv classes"""
		return set(self.root_size_map.keys())
