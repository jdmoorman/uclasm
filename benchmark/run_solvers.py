"""
run_solvers.py

Routines for running various other solvers.
"""

import os
import subprocess
import re


def run_LAD(tmplt, world, timeout=600, stop_first=False, induced=False,
            enum=False, verbose=False, keep_file=False):
    """
    This will run the LAD solver on the specified template and world graph.
    The remaining arguments adjust the operation of the LAD solver.
    Args:
        tmplt (Graph): The template graph
        world (Graph): The world graph
        timeout (int): The amount of time to run before quitting
        stop_first (bool): True if you want to stop after first isomorphism,
                           False if you want to count all isomorphisms
        induced (bool): True if you want induced subgraph isomorphism only
        enum (bool): True if you want to print all isomorphisms
        verbose (bool): True if you want verbose output
    Returns:
        A 4-tuple, the first is the number of isomorphisms found, second is
        number of bad nodes, third is number of nodes searched, 4th is
        the time taken.
    """

    # We need to create temporary files
    # We append the pid to prevent problems with multiprocessing
    tmp_tmplt_filename = 'template_' + str(os.getpid()) + '.txt'
    tmp_world_filename = 'world_' + str(os.getpid()) + '.txt'

    tmplt.write_file_solnon(tmp_tmplt_filename)
    world.write_file_solnon(tmp_world_filename)

    command = "pathLAD/main -p {} -t {} -s {}".format(tmp_tmplt_filename,
                                                      tmp_world_filename,
                                                      timeout)

    if stop_first:
        command += " -f"
    if induced:
        command += " -i"
    if enum:
        command += " -v"
    if verbose:
        command += " -vv"

    # We store the printed output of the program into a string
    output = subprocess.run(command, capture_output=True, shell=True, text=True)

    if not keep_file:
        os.remove(tmp_tmplt_filename)
        os.remove(tmp_world_filename)

    return parse_LAD_output(output.stdout)


def parse_LAD_output(output):
    """
    Parses the output from the Solnon Solver.
    TODO: get the candidates out of the LAD output.
    """
    return list(map(float, re.findall("[\d\.]+", output)))
