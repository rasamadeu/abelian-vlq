# abelian-vlq

A Python codebase to explore two Higgs doublets models (2HDMs) extended with isosinglets vector-like quarks (VLQs) and Abelian flavour symmetries.

## Introduction

This project was developed during my master's thesis, where I address a recent result in the first row of the Cabibbo-Kobayashi-Maskawa (CKM) matrix which
presents a deviation from the SM prediction of a unitary CKM. This result is known as the Cabibbo angle anomaly (CAA). 

The CAA is analysed in the context of a SM extension with a single up isosinglet VLQ. To reduce the number of parameters in the 
quark mass matrices, the maximally restrictive texture zero pairs (MRT) are determined. Maximally restrictive texture zero pairs are pairs of quark mass matrices
$(M_u, M_d)$ that have the maximal number of zero entries while maintaining compatibility with quark data (CKM and quark mass data).

The proposed mechanism to generate these texture zeros is to impose global Abelian symmetries on the fields.
The minimal setup for such mechanism consists of a 2HDM. For more details, read the thesis (Thesis.pdf) in the repository.

To determine the MRT realisable with 2HDM + Abelian symmetries, the following code pipeline was designed:

1. Determine the textures with most zero entries which generate non-zero CKM entries, $\gamma$ and quark masses (texture_zeros.py)
2. Determine the MRT by running a $\chi^2$ minimisation step to check if compatibility with data is attained (minimisation.py)
3. Determine the MRT that are realisable within the minimal 2HDM (abelian_symmetry_2HDM.py)

After concluding the thesis, steps **1** and **3** of the project pipeline were improved to deal with an arbitrary number of up and/or down isosinglet VLQs.

## Prerequisites

This project is written in **Python**.

There is a lot of information on
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

The repository contains a simple bash script that runs the pipeline explained in [Introduction]{# Introduction}.

## Issues

Currently, the minimisation.py script must be manually updated to perform the minimisation step in the procedure pipeline. It would be useful to automate this step.

This project is only prepared to deal with 2HDMs. However, there is no guarantee that for a particular set of values $(n_u, n_d)$ the MRT are realisable
with Abelian symmetries in the context of a 2HDM. It would be nice to allow for abelian_symmetry_2HDM.py to eventually determine the minimal NHDM that realizes a given texture.
