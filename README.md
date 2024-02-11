# A Lagrange Dual Learning Framework for Solving Constrained Inverse Kinematics Tasks

Project for MIT's 6.881 Optimization for Machine Learning course, Spring 2020.

![Inverse kinematics examples](/resources/images/ik1.png)

## Report and Video

See `report.pdf` for a project writeup and bibliography.

See https://www.youtube.com/watch?v=Ozp3JUacsZQ&feature=youtu.be for a quick presentation on the approach.

## Setup

### Dependencies

Simply run `pipenv install` from the top directory.

### Installing Drake on Mac

Depending on your environment, you may be able to simply install (Drake)[https://drake.mit.edu/installation.html] via `pip install drake`.

In my case on Mac M2, I had to install from a (downloaded archive)[https://drake.mit.edu/from_binary.html#stable-releases].

This repository uses **Drake** and **meshcat-python** for visualization.

Note that  **Drake is not needed for training models, only visualization and testing.**