# HVSeeker-proteins   
This tool is designed for training machine learning models on DNA sequence data and making predictions using pre-trained models. The tool supports various preprocessing methods, and allows for model training and prediction.
  
  

  
## Installation  
To set up your environment for this tool, follow these steps:  
  
**Clone the repository:**    

```
git clone https://github.com/BackofenLab/HVSeeker.git
cd HVSeeker/HVSeeker-Protein
```


  
**Install required Python using conda:**    


To install required python packages we recommend the use of miniconda


**Creating a Miniconda environment:**


First we install Miniconda for python 3. Miniconda can be downloaded from here:

https://docs.conda.io/en/latest/miniconda.html

Then Miniconda should be installed. On a linux machine the command is similar to this one:
```
bash Miniconda3-latest-Linux-x86_64.sh
```
Then we create an environment. The necessary setup is provided in the "environment.yml" file inside the "for_environment" directory

In order to install the corresponding environment one can execute the following command from the "for_environment" directory



```
conda env create -f HVSeeker_Prot_enviroment.yml --name HVSeekerProt
```

### Activation of the environment

Before running HVSeeker one needs to activate the corresponding environment.


```
conda activate HVSeekerProt
```

  
**Basic Usage HVSeeker-Proteins**  


Since HVSeeker-Proteins relies on ProtBert you will first have to clone the ProtBert github from here: https://github.com/nadavbra/protein_bert
```
git clone https://github.com/nadavbra/protein_bert.git --recurse-submodules
```

To run HVSeeker-Proteins you will also have to download the pretrained models from: 

https://drive.google.com/drive/folders/1wPgxfLnh-esQUB8xNhgnz9rJucmyX9Dm?usp=sharing

Then you can simply run the model using the following commands:



```
 python predict.py --evaluation True --output_path Sample_Data/neoag_val --test_file Sample_Data/neoag_val.csv --modelpath models/model.pkl```
