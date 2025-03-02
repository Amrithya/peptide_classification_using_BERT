
# Deepdefense: Automatic annotation and classification of immune systems in prokaryotes using Deep Learning


Deepdefense is a tool to serach for immune system cassettes based on the Doron system. To achieve this 
Deepdefense uses two distinct prediction modules. The first module consists of an ensemble of Deep Learning models to 
reject proteins, that do not belong to any immune system. The second module uses an ensemble of Deep Learning models
to classify proteins. We use a Deep Open Classifier (DOC) together with temperature scaling to calibrate our models.
The calibration allows us to either 1) reject proteins 2) classifiy them as a known type or 3) classify them as
an unknown, potentially new type of protein.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

First you need to install Miniconda
Then create an environment and install the required libraries in it


### Creating a Miniconda environment 

First we install Miniconda for python 3.
Miniconda can be downloaded from here:

https://docs.conda.io/en/latest/miniconda.html 

Then Miniconda should be installed. On a linux machine the command is similar to this one: 

```
bash Miniconda3-latest-Linux-x86_64.sh
```

Then we create an environment. The necessary setup is provided in the "environment.yml" file inside the "for_environment" directory

In order to install the corresponding environment one can execute the following command from the "for_environment" directory

```
conda env create -f environment.yml --name deepdef
```



### Activation of the environment

Before running DeepDefense one need to activate the corresponding environment.

```
conda activate deepdef
```



## Running Deepdefense


### Models


Due to the file size restrictions of github, models are available on:

https://drive.google.com/file/d/1qKGUu8P143yezrX0nCFHh25vBLTTmtJ3/view?usp=share_link
https://drive.google.com/file/d/1Obek8fj2G67UeDVN-95Em_msagcgb1qO/view?usp=sharing

Put them in a models folder and extract the models.

### Training custom models

```
python main_binary.py

```

