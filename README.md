# abelian-vlq

A Python codebase to explore two Higgs doublets models (2HDMs) extended with isosinglets vector-like quarks (VLQs) and Abelian flavour symmetries.

## Introduction

This project was developed during my master's thesis, where I address a recent development in the measurement of the first row of the Cabibbo-Kobayashi-Maskawa (CKM) matrix. This
result, known as the Cabibbo angle anomaly (CAA), presents a deviation from the SM prediction of a unitary CKM. 

The CAA is analysed in the context of a SM extension with a single up isosinglet VLQ. To reduce the number of parameters in the 
quark mass matrices, the maximally restrictive texture zero pairs (MRT pairs) are determined.

MRT pairs are pairs of quark mass matrices
$(M_u, M_d)$ that have the maximal number of zero entries while maintaining compatibility with quark data (CKM and quark mass data).

The proposed mechanism to generate these texture zeros is to impose global Abelian symmetries on the fields.
The minimal setup for such mechanism consists of a 2HDM. For more details, check the [thesis](Thesis.pdf) in the repository.

To determine the MRT realisable with 2HDM + Abelian symmetries, the following code pipeline was designed:

1. Determine the textures with most zero entries which generate non-zero CKM entries, $\gamma$ and quark masses - `texture_zeros.py`
2. Determine the MRT by running a $\chi^2$ minimisation step to check if compatibility with data is attained - `minimisation.py`
3. Determine the MRT that are realisable within the minimal 2HDM - `abelian_symmetry_2HDM.py`

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

The Python modules are in the **src** directory. The repository contains a simple bash script that runs the pipeline explained in [Introduction](#Introduction).
To run it, you simply need to type

`./abelian-vlq n_u n_d`

in the terminal, where n_u and n_d are the number of up and down isosinglet VLQs, respectively. The script results are written in a created folder **output**.

> [!IMPORTANT]
> Remember to modify minimisation.py to accomodate the case you wish to study, since this step must be manually defined.

### Conditions

There are several conditions which you can redefine in the scripts to accomodate your demands: 

- `texture_zeros.py`
    - In line 202 there is a block of code where extra compatibility criteria may be defined.
    There, you find the criteria that at least 1 VLQ must be coupled to the SM quarks. Otherwise, the VLQs would be decoupled and the CKM matrix would remain unitary.
- `minimisation.py`
    - There are global constants that control the minimisation parameters:
        - **N_TRIES** - specifies how many times the minimisation step is performed on each texture pair before it is ruled out as incompatible with data
        - **MAX_CHI_SQUARE** - specifies the allowed $\chi^2$ upper bound for each observable
        - **VLQ_LOWER_BOUND** - specifies the allowed lower bound for the VLQs mass
        - **SIGMA_MASS_VLQ** - specifies the estimated error associated with VLQ_LOWER_BOUND
        - **FILENAME_OUTPUT** - if provided, specifies filename for output
- `abelian_symmetry_2HDM.py`
    - There is a single parameter, **FILENAME_OUTPUT**, which is similar to the previous case.

### Useful scripts

`io_mrt.py` is this project I/O Python module which contains function to I/O data from/to files after each step of the pipeline. It also contains pretty print functions which writes files that present data in a more readable way.

Finally, `notation.py` is module which contains functions that translates data from each step into tables in Latex format, reducing the workload and human error when transcribing the results to Latex documents. It contains the following functions:

- `write_table_notation`: writes table that automatically associates a label with each mass matrix texture for up and down quarks 
- `write_table_pairs`: writes a table with all MRT pairs obtained from steps **1** and **2**, using the notation defined in `write_table_notation`
- `write_table_charges`: writes a table with the field charges for the MRT pairs imposed by 2HDM + Abelian symmetries (end result of the pipeline). Also defines a notation for the possible set of charges for each MRT pair.
- `write_table_decomp`: writes a table with the Yukawa matrices decomposition corresponding to each field charges set. The non-zero entries are labeled as follows:
    - 1 - non-zero entry from Yukawa coupling to the first Higgs doublet
    - 2 - non-zero entry from Yukawa coupling to the second Higgs doublet
    - 3 - non-zero entry from bare mass term with VLQ

> [!NOTE]
> The module `abelian_symmetry_2HDM.py` only provides a numerical solution to the field charges. Hence, the charges in the table resulting from `write_table_charges` module may have floating point values. It is the user's responsibility to analyse the results and rewrite them in a more convenient manner.

## Issues

Currently, the `minimisation.py` script must be manually updated to perform the minimisation step in the procedure pipeline. It would be useful to automate this step.

This project is only prepared to deal with 2HDMs. However, there is no guarantee that for a particular set of values $(n_u, n_d)$ the MRT are realisable
with Abelian symmetries in the context of a 2HDM. It would be nice to allow for `abelian_symmetry_2HDM.py` to eventually determine the minimal NHDM that realizes a given texture.

## To do

- [] Improve `get_non_restrictive_texture_zeros()` to deal with textures of an arbitrary dimension
- [] Automate `minimisation.py` to deal with an arbitrary number of up and or down VLQs
- [] Improve `abelian_symmetry_2HDM.py` to determine minimal NHDM that realizes a given MRT pair
