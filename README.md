
# AML-Project
**Virginia Tech** Fall 2019

**Advanced Machine Learning** Course Project

by *Adheesh Juvekar, Kunal Joshi, Rahul Iyer*


## Our Job
We aim to reproduce the research done by Diederik et al. in Adam[^1] paper. We are testing the results obtained by them on CIFAR dataset with a different CNN architecture. More details to be added.

## Getting Started
Here's how you can setup and run this project

### Prerequisites
What things you need to install
```
Anaconda3 [recommended]
Python=3.5.x
```

### Installing
Create virtual environment
```
conda env create -f environment.yml
```
or
```
conda create --name <env_name> python=3.5 --file requirements.txt
```
NOTE: Replace <env_name> with your choice of environment name or change name in environment.yml file if using .yml file.

### Activate
Activate the environment
```
conda activate env_name
```
use amlproj as env_name if directly installed using .yml file.

### Run the file as
```
python main.py
```

[^1]:  [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
