# A Lagrange Dual Learning Framework for Solving Constrained Inverse Kinematics Tasks

Project for MIT's 6.881 Optimization for Machine Learning course, Spring 2020.

![Inverse kinematics examples](/resources/images/ik1.png)

## Report and Video

See `report.pdf` for a project writeup and bibliography.
See https://www.youtube.com/watch?v=Ozp3JUacsZQ&feature=youtu.be for a quick presentation on the approach.

## Disclaimers

- This project has only been tested with `Python3.7` and `torch==1.4.0` on Ubuntu 18.04!
- Also, there a lot of hard-coded absolute paths to folders on my computer, so things probably won't run out-of-the-box without some modifications... sorry!

## Setup

### (Optional) Install PyDrake
I use **Drake** and **meshcat-python** for visualization. **They aren't needed for training models, so you can skip these steps if just using the training code.**

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
