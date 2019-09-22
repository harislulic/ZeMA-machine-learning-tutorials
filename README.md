# Click to try notebooks on MyBinder.org
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/harislulic/ZeMA-machine-learning-tutorials/master)
# ZeMA machine learning tutorials (with incorporated uncertainties WIP)
These notebooks were developed on the basis of previous work on Machine Learning methods applied on ZeMA-s testbed data of the co-author [Haris Lulic](mailto:haris.lulic@met.gov.ba). The original notebooks can be approached in branch [computation_without_uncertaintites](https://github.com/harislulic/ZeMA-machine-learning-tutorials/tree/computation_without_uncertaintites).

Machine Learning tutorials oriented at begginers in data science. Methods are applied on ZeMA-s testbed data (Zentrum für Mechatronik und Automatisierungstechnik gGmbH). 

Get started
---
Clone the repository to your local machine using instructions from [here](https://help.github.com/en/articles/cloning-a-repository).

Anaconda and Python Installation
---
If you don't have *Anaconda* installed already follow [this guide
](https://jupyter.readthedocs.io/en/latest/install.html#installing-jupyter-using-anaconda-and-conda). Anaconda conveniently installs Python, the Jupyter Notebook, and other commonly used packages for scientific computing and data science. Notebooks presented here will also require installation of pip and package PyDynamic. Activate your Anaconda's environment in the command prompt and write:
```
conda install pip
```
and then:
```
pip install PyDynamic
```
For interactive diagrams, activate your Anaconda's environment in the command prompt and write:
```
pip install ipywidgets
```
and then:
```
jupyter nbextension enable --py widgetsnbextension
```
