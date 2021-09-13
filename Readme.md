Code for the paper _A Biological Neural Network Model For Cognitive Processes Using Graph Traversal_ 
====================================================================================================

# Running the code
On your commandline, just invoke `python Hybrid_Neuron_Simulation.py` to run an example simulation.
Results will be visualized in several plot windows, that will be live updated.
Frames and videos will also be stored directly in the working directory.

# Simulation Setups
The different setups are defined in the file `setups.py`, located in the root directory of the repository.
To run the simulation with a specific setup, e.g. `complex_maze`, simply add the setup name as command line parameter, e.g. `python Hybrid_Neuron_Simulation.py complex_maze`.
In case of supplying an invalid setup name, a list of available setups will be printed.

To run simulations for multiple different setups sequentially, you can use something along the lines of `for setup in empty simple s_maze central_block_homogeneous central_block complex_maze; do python Hybrid_Neuron_Simulation.py $setup; done`.
