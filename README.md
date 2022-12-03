### Reinforcement Learning Toolbox for Robotics

Toolbox with highly optimized implementations of deep reinforcement learning algorithms for robotics using Pytorch and Python.

In this project, two colleagues and I develop a toolbox with state of the art reinforcement learning algorithms using Pytorch and Python. The toolbox contains other useful features such as custom robotics environments, loggers, plotters and much more.

## About The Project
### Toolbox Structure
All the algorithms are in the `rlbotics` directory. Each algorithm specified above has an individual directory.

### Common
The directory `common` contains common modular classes to easily build new algorithms.
- `approximators`: Basic Deep Neural Networks (Dense, Conv, LSTM).
- `logger`: Log training data and other information
- `visualize`: Plot graphs
- `policies`: Common policies such as Random, Softmax, Parametrized Softmax and Gaussian Policy
- `utils`: Functions to compute the expected return, the Generalized Advantage Estimation (GAE), etc.

### Algorithm Directories
Each algorithm directory contains at least 3 files:
- `main.py`: Main script to run the algorithm
- `hyperparameters.py`: File to contain the default hyperparameters
- `<algo>.py`: Implementation of the algorithm
- `utils.py`: (Optional) File containing some utility functions

Some algorithm directories may have additional files specific to the algorithm.

## Contributing
To contribute to this package, it is recommended to follow this structure:
- The new algorithm directory should at least contain the 3 files mentioned above.
- `main.py` should contain at least the following functions:
  - `argparse`: Parses input argument and loads default hyperparameters from `hyperparameter.py`.
  - `main`: Parses input argument, builds the environment and agent, and train the agent.
  - `train`: Main training loop called by main()

- `<algo>.py` should contain at least the following methods:
  - `__init__`: Initializes the classes
  - `_build_policy`: Build policy
  - `_build_value_function`: Build value function
  - `compute_policy_loss`: Build policy loss function
  - `update_policy`: Update the policy
  - `update_value`: Update the value function

## Getting Started

### Prerequisites
* The program was created using **Python3.7**
* Pytorch
* Numpy
* Pandas
* Tensorboard
* Seaborn
* Scipy
* Gym

### Installation
To install the RLBotics toolbox, install the required librarires and clone this repository using the following commands:

```
pip install torch
pip install numpy
pip install pandas
pip install tensorboard
pip install seaborn
pip install gym
git clone https://github.com/kousheekc/Reinforcement-Learning-Toolbox-for-Robotics.git
```

## Usage
To run the an algorithm on a perticular environment, open a terminal and navigate to the folder you just cloned and run the following command:
```
python3 -m rlbotics.algo.main
```
Where `algo` can be replaced by the algorithm you wish to use. You can also pass in arguments, or modify the `hyperparameters.py` file contained in each algorithm folder to change the environment and other hyperparameters related to the algorithm.

Once the algorithm is running you can deploy a tensorboard session ot track the progress.

## License
Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## Contact
Kousheek Chakraborty - kousheekc@gmail.com

Dyuman Aditya - dyuman2001@gmail.com

Suman Pal - suman7495@gmail.com

Project Link: [https://github.com/kousheekc/Reinforcement-Learning-Toolbox-for-Robotics](https://github.com/kousheekc/Reinforcement-Learning-Toolbox-for-Robotics)

