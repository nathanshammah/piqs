## Installation
### Linux and MacOS
We have made the package available in [conda-forge](https://conda-forge.org/). Download and install Anaconda or Miniconda from [https://www.anaconda.com/download/](https://www.anaconda.com/download/) and then install `piqs` from the conda-forge repository with,
```
conda config --add channels conda-forge
conda install conda-forge piqs
```
Then you can fire up a Jupyter notebook and run `piqs` out of your browser.

### Windows
We will add a Windows version soon but if you are on Windows, you can build piqs from source by first installing conda. You can add the conda-forge channel with `conda config --add channels conda-forge`. Then, please install `cython`, `numpy`, `scipy` and `qutip` as `piqs` depends on these packages. The command is simply,

```conda install cython numpy scipy qutip```

Finally, you can download the [source code](https://github.com/nathanshammah/piqs/archive/v1.2.tar.gz), unzip it and run the setup file inside with,
```
python setup.py install
```
If you have any problems installing the tool, please open an issue or write to us.