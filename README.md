# Instrument simulator for ZShooter
This repository contains notebooks and code for simulating ZShooter using ScopeSim.

The ScopeSim and irdb packages installed through zs-scopesim (pyproject.toml) are forked 
versions with additional custom features needed for ZShooter.

Additionally, a modified version of [PALACE code ](https://zenodo.org/records/14064023) is used by 
the modified ScopeSim for modeling sky line-emission background component. This code is provided
in the `PALACE/palace` directory and will be installed as a package when installing zs-scopesim. 

## Installation
Clone the repository:
```bash
git clone https://github.com/CaltechOpticalObservatories/zs-scopesim.git
cd zs-scopesim
```
Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate 
```
Install the package:
```bash
pip install -e .
```
In case there are issues in PALACE installation due to the Cython build, use:
```bash
pip install -e . --no-build-isolation
```

