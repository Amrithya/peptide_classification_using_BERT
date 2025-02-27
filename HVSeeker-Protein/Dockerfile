# Use a CUDA base image with CUDA 11.0 and cuDNN 8.0 (compatible with TensorFlow 2.4)
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

# Set working directory
WORKDIR /app

# Install Python, pip, and wget
RUN apt-get update && apt-get install -y python3-pip wget bzip2 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*


RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh
    
# Add Conda to the PATH
ENV PATH=/opt/conda/bin:$PATH

# Copy and install your custom dependencies
COPY environment.yml .
#RUN pip install --upgrade pip && conda env create -f environment.yml
RUN conda create -n hvseekerProt python=3.8
#RUN  conda env create -f environment.yml
#RUN conda run -n hvseekerProt pip install tensorflow==2.4.0

RUN wget https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-2.4.0-cp38-cp38-manylinux2010_x86_64.whl && \
    conda run -n hvseekerProt pip install tensorflow_gpu-2.4.0-cp38-cp38-manylinux2010_x86_64.whl && \
    rm tensorflow_gpu-2.4.0-cp38-cp38-manylinux2010_x86_64.whl



# Copy the environment.yml file
COPY environment.yml .

# Install the dependencies from environment.yml into the Conda environment
RUN conda run -n hvseekerProt pip install scikit-learn==1.2.2
RUN conda run -n hvseekerProt pip install scikit-optimize==0.9.0
RUN conda run -n hvseekerProt pip install tensorflow-addons==0.12.1
RUN conda run -n hvseekerProt pip install numpy==1.20.1
RUN conda run -n hvseekerProt pip install pandas==1.2.3
RUN conda run -n hvseekerProt pip install h5py==3.2.1
#RUN conda run -n hvseekerProt pip install lxml==4.3.2
RUN conda run -n hvseekerProt pip install pyfaidx==0.5.8



#RUN conda env update -n hvseekerProt -f environment.yml

# Upgrade pip to the latest version
# RUN pip install --upgrade pip

# Install TensorFlow 2.4 with GPU support
# RUN pip install tensorflow-gpu==2.4.0

# Copy your Python scripts
ADD protein_bert protein_bert
COPY optimize_finetuning.py .
COPY predict.py .
COPY train.py .

# Run your Python script
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "hvseekerProt"]
CMD ["python -u"]
