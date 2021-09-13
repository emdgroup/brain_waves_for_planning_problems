Code for Simulations in the Publication _Can the brain use waves to solve planning problems?_
=============================================================================================

# Installing Required Python Packages
Please use Python version 3.6 at least.
Required packages are listed in `requirements.txt` and can be installed using `pip` via `pip install -r requirements.txt`.
Furthermore, you need to install `ffmpeg` and add its binary directory to your `$PATH` environment variable.


# Installing Required Python Packages
Please use Python version 3.6 at least.
Required packages are listed in `requirements.txt` and can be installed using `pip` via `pip install -r requirements.txt`.
# Running the code
On your commandline, just invoke `python Hybrid_Neuron_Simulation.py` to run an example simulation.
Results will be visualized in several plot windows, that will be live updated.
Frames and videos will also be stored directly in the working directory.

# Simulation Setups
The different setups are defined in the file `setups.py`, located in the root directory of the repository.
To run the simulation with a specific setup, e.g. `complex_maze`, simply add the setup name as command line parameter, e.g. `python Hybrid_Neuron_Simulation.py complex_maze`.
In case of supplying an invalid setup name, a list of available setups will be printed.

To run simulations for multiple different setups sequentially, you can use something along the lines of `for setup in empty simple s_maze central_block_randomized central_block complex_maze; do python Hybrid_Neuron_Simulation.py $setup; done`.

# License
Copyright (c) 2021 Merck KGaA, Darmstadt, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The full text of the license can be found in the file [LICENSE](LICENSE) in the repository root directory.

# Contributing
Contributions to the package are always welcome and can be submitted via a pull request.
Please note, that you have to agree to the [Contributor License Agreement](CONTRIBUTING.md) to contribute.
