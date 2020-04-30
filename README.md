# A lagrange dual approximation for inverse kinematics with constraints

Project for 6.881 Spring 2020 (Optimization for Machine Learning)

## Disclaimers

- This project has only been tested with `Python3.7` and `torch==1.4.0` on Ubuntu 18.04!
- Also, there a lot of hard-coded absolute paths to folders on my computer, so things probably won't run out-of-the-box without some modifications... sorry!

## Setup

### (Optional) Install PyDrake
I use Drake and meshcat-python for visualization. **They aren't needed for training models, so you can skip these steps if just using the training code.**

Installing [PyDrake with Python bindings](https://drake.mit.edu/python_bindings.html):
```bash
git clone https://github.com/RobotLocomotion/drake.git
mkdir drake-build
cd drake-build
cmake ../drake
make -j
```

Add this to `.bashrc` to help Python find Drake:
```bash
cd drake-build
export PYTHONPATH=${PWD}/install/lib/python3.6/site-packages:${PYTHONPATH}
```

### Install training requirements
Also run `pip install -r requirements37.txt` to install other Python packages.
