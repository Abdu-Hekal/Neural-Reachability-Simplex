# Neural-Reachability-Simplex
## Overview

This repository employs neural reachability in a simplex architecture, where a neural reachability module acts as an online monitoring system which switches back and forth between an unverified, advanced controller and a safe, baseline controlle. The framework is
deployed in a simulated environment on the [f1tenth gym](https://f1tenth-gym.readthedocs.io/en/latest/) platform, where an autonomous vehicle aims to drive around a track efficiently and safely, while avoiding static obstacles. To see neural reachability employed as a monitoring system for a vehicle governed by an MPC controller and completing a set of maneuvers, see the repository [here](https://github.com/Abdu-Hekal/Neural-Reachability).

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Cititation](#Citation)
- [Installation Issues](#installation-issues)

## Introduction

Neural-Reachability is a framework for neural network-based reachability analysis. We train deep neural networks to learn the reachable sets of dynamical systems. Formally, we say that given a system, a set of initial conditions, a disturbance set, and a bounded time interval of interest, neural reachability estimates the reachable sets of the system through deep neural networks. In this repository, we deploy neural reachability in a simplex architecture as an online decision module for an autonomous vehicle racing around a track, whilst avoiding static obstacles. The advanced MPC controller is given default control and neural reachability continuously checks the safety of each command sequence by approximating the corresponding reachable set. If a potential collision with an obstacle is detected, it switches control to the Follow the Gap method which retains control until a command sequence of the MPC controller is deemed safe. Once this holds, control of the vehicle is switched back to the MPC controller to follow the optimal trajectory to drive around the track.

## Prerequisites

- **Python(<3.12)** - Install Python strictly less than version 3.12.


## Installation

Clone the repository:

```bash
git clone https://github.com/Abdu-Hekal/Neural-Reachability-Simplex.git
```

Install the f1tenth_gym environment shipped with the repository:

```bash
cd Neural-Reachability-Simplex/f1tenth_gym
pip install .
cd ..
```

Install requirements:

```bash
pip install -r requirements.txt
```


## Usage

From the terminal, run:

```bash
python simplex.py
```

to render an f1tenth simulation of vehicle(s) racing around a track with static obstacles. Each vehicle has a simplex controller governed by a neural reachability decision module.

Additionally, `simplex.py` supports the following command-line arguments:

- `-z` or `--zoom`: Zoom in the camera.
- `-n <number>` or `--number <number>`: Number of vehicles (default is 1).
- `-o <obstacles>` or `--obstacles <obstacles>`: Number of obstacles (default is 5).

## Usage

See the following example visualizing a vehicle avoiding an obstacle detected by the neural reachability module. Once the predicted reachable sets of the system (green) intersect with the obstacle, the decision module switches from the advanced model predictive controller, to the base controller (Follow the Gap).

![](https://github.com/Abdu-Hekal/Neural-Reachability-Simplex/blob/main/f110_simplex.gif)



## Citation

This work on Neural reachability has been published at the 2022 IEEE 25th International Conference on Intelligent Transportation (ITSC), available [here](https://ieeexplore.ieee.org/abstract/document/9922294).

If you cite this work, please cite

Bogomolov, S., Hekal, A., Hoxha, B. and Yamaguchi, T., 2022, October. Runtime Assurance for Autonomous Driving with Neural Reachability. In 2022 IEEE 25th International Conference on Intelligent Transportation Systems (ITSC) (pp. 2634-2641). IEEE.

Bibtex:
```
@inproceedings{bogomolov2022runtime,
  title={Runtime Assurance for Autonomous Driving with Neural Reachability},
  author={Bogomolov, Sergiy and Hekal, Abdelrahman and Hoxha, Bardh and Yamaguchi, Tomoya},
  booktitle={2022 IEEE 25th International Conference on Intelligent Transportation Systems (ITSC)},
  pages={2634--2641},
  year={2022},
  organization={IEEE}
}
```

## Installation Issues

There are two common installation issues, the first related to the `pycddlib` library required by `pytope` and the second related to rendering with `f1tenth_gym`

### pycddlib

When installing requirements which installs `pycddlib`, a common issue arises when the installation process fails to locate the `gmp.h` header file. This file is essential as it contains declarations necessary for using the GNU Multiple Precision Arithmetic Library (GMP). 

To resolve this issue, you can set environment variables that point the compiler (`clang` or `gcc`) to the directory where `gmp.h` is located. Here's how you can do it:

1. **Check GMP Installation:**
   First, ensure that GMP is installed on your system. You can install it using package managers like Homebrew on macOS or `apt` on Ubuntu/Debian:
   
   **On macOS:**
   ```bash
   brew install gmp
   ```

   **On Ubuntu/Debian:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y libgmp-dev
   ```

2. **Set Environment Variables:**
   Once GMP is installed, export the paths to the GMP header files (gmp.h) and libraries (libgmp.a or libgmp.so). This tells the compiler where to find these files during the build   process:
   
   **On macOS:**
   ```bash
   export C_INCLUDE_PATH=$(brew --prefix gmp)/include
   export LIBRARY_PATH=$(brew --prefix gmp)/lib
   ```

   **On Ubuntu/Debian:**
   ```bash
   export C_INCLUDE_PATH=/usr/include/
   export LIBRARY_PATH=/usr/lib/
   ```

4. **Reinstall pycddlib:**
   After setting the environment variables, attempt to install pycddlib again using pip:
   ```bash
   pip install pycddlib
   ```

### f1tenth_gym
   
Another common installation issue is related to installation of f1tenth_gym.
On MacOS Big Sur and above, you may encounter this error:

```bash
ImportError: Can't find framework /System/Library/Frameworks/OpenGL.framework.
```

or 

```bash
AttributeError: 'Batch' object has no attribute 'add'
```


The error can be fixed by installing this version of pyglet:
```bash
pip install pyglet==1.5.20
```



   
    

