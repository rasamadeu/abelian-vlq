# abelian-vlq

A Python codebase to explore 2HDM extended with isosinglets vector-like quarks (VLQ) and Abelian flavour symmetries.

## Introduction



## Prerequisites

This project is written in **Python**. There is a lot of information on
how to install the Python interpreter online. In my case, since I'm working with Ubuntu, I installed it via apt by typing in the terminal:

```
sudo apt update
sudo apt install python3
```

The project is also dependent on the following open source Python packages, which are very useful for scientific computing:

- [**NumPy**](https://numpy.org/install/) - standard library used for scientific computing, providing a high level of abstraction with well-optimized C code
- [**Numba**](https://numba.readthedocs.io/en/stable/user/installing.html) - open source Python just-in-time (JIT) compiler
- [**iMinuit**](https://scikit-hep.org/iminuit/install.html) - a Python interface for the **Minuit2** C++ library maintained by CERN'S ROOT team.

## Usage

## Issues

Currently, the minimisation.py script must be manually updated to perform the minimisation step in the procedure pipeline. It would be useful to automate this step.
