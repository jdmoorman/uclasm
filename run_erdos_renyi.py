import uclasm

from timeit import default_timer
from time import sleep
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix

from multiprocessing import Process, Queue

np.random.seed(0)
timeout = 1000

def process_fn(tmplt, world, result_queue=None, label=None, count_isomorphisms=False):
	result = {}
	result["label"] = label # For identifying results afterwards
	start_time = default_timer()
	tmplt, world, candidates = uclasm.run_filters(tmplt, world, candidates=tmplt.is_cand, filters=uclasm.cheap_filters, verbose=False)
	filter_time = default_timer()-start_time
	# print("Time taken for filters: {}".format(filter_time))
	# filter_times.append(filter_time)
	result["filter_time"] = filter_time

	# start_time = default_timer()
	# from filters.validation_filter import validation_filter
	# validation_filter(tmplt, world, candidates=candidates, in_signal_only=False,
	#       verbose=False)
	# print("Time taken for validation: {}".format(default_timer()-start_time))
	# validation_times += [default_timer()-start_time]
	# # tmplt.candidate_sets = {x: set(world.nodes[candidates[idx,:]]) for idx, x in enumerate(tmplt.nodes)}

	if count_isomorphisms:
		# # print("Starting isomorphism count")
		start_time = default_timer()
		count, n_iterations = uclasm.count_isomorphisms(tmplt, world, candidates=candidates, verbose=False, count_iterations=True)
		# print("Counted {} isomorphisms in {} seconds".format(count, default_timer()-start_time))
		iso_count_time = default_timer() - start_time
		# iso_counts += [count]
		# iso_count_times += [default_timer()-start_time]
		result["n_isomorphisms"] = count
		result["iso_count_time"] = iso_count_time
		result["has_iso"] = count > 0
	else:
		start_time = default_timer()
		from uclasm.counting.has_isomorphism import has_isomorphism
		has_iso, n_iterations = has_isomorphism(tmplt, world, candidates=candidates, verbose=False, count_iterations=True)
		# if has_iso:
		# 	print("Has isomorphism")
		# else:
		# 	print("No isomorphism")
		iso_check_time = default_timer() - start_time
		# print("Isomorphism checked in {} seconds".format(iso_check_time))
		# iso_check_times.append(iso_check_time)

		result["iso_check_time"] = iso_check_time
		result["has_iso"] = has_iso
	result["n_iterations"] = n_iterations

	if result_queue is not None:
		result_queue.put(result)
	else:
		return result

def run_trial(n_tmplt_nodes, n_world_nodes, n_layers, tmplt_prob, world_prob, results, use_timeout=True, count_isomorphisms=False):
	run_process = None
	try:
		if use_timeout:
			result_queue = Queue()

			run_process = create_process(n_tmplt_nodes, n_world_nodes, n_layers, tmplt_prob, world_prob, count_isomorphisms=count_isomorphisms)
			run_process.start()

			start_time = default_timer()
			while run_process.is_alive() and default_timer() - start_time < timeout:
				sleep(0.5)

			if run_process.is_alive():
				print("Timeout exceeded, killing process")
				run_process.terminate()
			else:
				result = result_queue.get()
				result['n_tmplt_nodes'] = n_tmplt_nodes
				result['n_world_nodes'] = n_world_nodes
				result['tmplt_prob'] = tmplt_prob
				result['world_prob'] = world_prob
				result['n_layers'] = n_layers
				if use_timeout:
					result['timeout'] = timeout
				# print(result)
				results.append(result)
		else:
			tmplt, world = make_graphs(n_tmplt_nodes, n_world_nodes, n_layers, tmplt_prob, world_prob)
			result = process_fn(tmplt, world, label=(tmplt_prob, world_prob), count_isomorphisms=count_isomorphisms)
			result['n_tmplt_nodes'] = n_tmplt_nodes
			result['n_world_nodes'] = n_world_nodes
			result['tmplt_prob'] = tmplt_prob
			result['world_prob'] = world_prob
			result['n_layers'] = n_layers
			if use_timeout:
				result['timeout'] = timeout
			# print(result)
			results.append(result)

	except KeyboardInterrupt:
		print("Interrupting process")
		if run_process is not None and run_process.is_alive():
			run_process.terminate()
		raise KeyboardInterrupt

def create_process(n_tmplt_nodes, n_world_nodes, n_layers, tmplt_prob, world_prob, result_queue, count_isomorphisms=False):
	tmplt, world = make_graphs(n_tmplt_nodes, n_world_nodes, n_layers, tmplt_prob, world_prob)
	run_process = Process(target=process_fn, args=(tmplt, world), kwargs={"result_queue": result_queue, "label": (tmplt_prob, world_prob), "count_isomorphisms": count_isomorphisms})
	return run_process

def make_graphs(n_tmplt_nodes, n_world_nodes, n_layers, tmplt_prob, world_prob):
	tmplt_nodes = [x for x in range(n_tmplt_nodes)]
	world_nodes = [x for x in range(n_world_nodes)]

	tmplt_shape = (n_tmplt_nodes, n_tmplt_nodes)
	world_shape = (n_world_nodes, n_world_nodes)

	tmplt_adj_mats = [csr_matrix(np.random.choice([0, 1], tmplt_shape, p=[1-tmplt_prob, tmplt_prob])) for i in range(n_layers)]
	world_adj_mats = [csr_matrix(np.random.choice([0, 1], world_shape, p=[1-world_prob, world_prob])) for i in range(n_layers)]

	channels = [str(x) for x in range(n_layers)]

	tmplt = uclasm.Graph(np.array(tmplt_nodes), channels, tmplt_adj_mats)
	world = uclasm.Graph(np.array(world_nodes), channels, world_adj_mats)

	tmplt.is_cand = np.ones((tmplt.n_nodes,world.n_nodes), dtype=np.bool)
	tmplt.candidate_sets = {x: set(world.nodes) for x in tmplt.nodes}

	return tmplt, world

# n_tmplt_nodes = 10
n_world_nodes = 150
# n_layers = 1
n_trials = 40
n_cores = 40
count_isomorphisms = False

n_tmplt_nodes = 10
tmplt_prob = 0.5
world_prob = 0.5

# for n_layers in [1,3,5,7,9]:
n_layers = 1
if True:
	results = []
	import tqdm
	for n_world_nodes in tqdm.tqdm(range(10, 300, 5)):
		n_trials_remaining = n_trials
		while n_trials_remaining > 0:
			process_list = []
			result_queue = Queue()
			n_processes = n_cores if n_cores < n_trials_remaining else n_trials_remaining
			for i in range(n_processes):
				# print("Creating process {}".format(i))
				# run_trial(n_tmplt_nodes, n_world_nodes, n_layers, tmplt_prob, world_prob, results, use_timeout=True)
				new_process = create_process(n_tmplt_nodes, n_world_nodes, n_layers, tmplt_prob, world_prob, result_queue, count_isomorphisms=count_isomorphisms)
				process_list.append(new_process)
				new_process.start()
			start_time = default_timer()
			n_finished = n_processes
			while default_timer() - start_time < timeout:
				any_alive = False
				for process in process_list:
					if process.is_alive():
						any_alive = True
				if not any_alive:
					break
				sleep(0.5)
			for process in process_list:
				if process.is_alive():
					process.terminate()
					n_finished -= 1
			for i in range(n_finished):
				result = result_queue.get()
				result['n_tmplt_nodes'] = n_tmplt_nodes
				result['n_world_nodes'] = n_world_nodes
				result['tmplt_prob'] = tmplt_prob
				result['world_prob'] = world_prob
				result['n_layers'] = n_layers
				results.append(result)
			n_trials_remaining -= n_processes

		np.save("erdos_renyi_results_{}_trials_{}_layers{}_timeout_{}_vary_world_size".format(n_trials, n_layers, "_count_iso" if count_isomorphisms else "", timeout), results)
