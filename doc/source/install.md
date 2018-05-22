## Installation
### Linux and MacOS
We have made the package available in [conda-forge](https://conda-forge.org/). Download and install Anaconda or Miniconda from [https://www.anaconda.com/download/](https://www.anaconda.com/download/) and then install `piqs` from the conda-forge repository with
```
conda config --add channels conda-forge
conda install conda-forge piqs
```
Then you can fire up a Jupyter notebook and run `piqs` out of your browser.

### Windows
We will add a Windows version soon but if you are on Windows, you can build piqs from source by first installing conda. If you have a python environment with `cython`, `numpy`, `scipy` and `qutip` installed then simply download the [source code](https://github.com/nathanshammah/piqs/archive/v1.2.tar.gz), unzip it and run the setup file inside with

```
python setup.py install
```
If you have any problems installing the tool, please open an issue or write to us.

### Source
Alternatively, you can download the package from the [source](https://github.com/nathanshammah/piqs/archive/v1.2.tar.gz) and install with

```
python setup.py install
```