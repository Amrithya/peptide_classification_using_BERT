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
python predict.py --output_path {prefix of output file} --test_file {test_file} --modelpath {path_to_model}
```
If you want to run evaluation instead, simply add the --evaluation flag. 

additionally we provide a file for optimizing and training on a novel dataset:

```
python optimize_finetuning.py -o {output_file}  -f {file_for_finetuning}
```
and for training:

```
train.py -t {test_file} -f {training_file}
```

**Using docker**

Alternatively we provide a docker container capable of running HVSeeker_Protein. You can build it using the Dockerfile provided in the docker folder.
First move the Dockerfile and enviroment.yml into the HVSeeker_Protein main directory. Make sure you have docker and and nvidia-container-toolkit installed:

```
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

then build the docker image:
goo```
sudo docker build --network host -t hvseekerprot .

```
and run with for training:

```
sudo docker run --network host --gpus all --privileged=true -v {your-path}:/app/input -v {your-path-to-models}:/app/models hvseekerprot  bash -c "yes 'Yes' |  python -u train.py -t input/{data} -f input/{data} -s models"
```

or prediction:

```
sudo docker run --network host --gpus all -v {your-path}:/app/input -v {your-path-to-models}:/app/models hvseekerprot  bash -c "yes 'Yes' |  python -u predict.py --test_file input/{test_file} --output_path input/{output name} -m models/{model_name}
```

if you want to evaluate the quality of the prediction simply add the --evaluation Flag

```
sudo docker run --network host --gpus all -v {your-path}:/app/input -v {your-path-to-models}:/app/models hvseekerprot  bash -c "yes 'Yes' |  python -u predict.py --evaluation --test_file input/{test_file} --output_path input/{output name} -m models/{model_name}
```

We made the build image available here: https://drive.google.com/file/d/1Bf3FMoI2rCDROma13lLfID1AaoeAvJ7L/view?usp=sharing

### Output

The script predict.py creates a table that summarizes the sequence, label and prediction 



