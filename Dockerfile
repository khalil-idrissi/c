FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# Set the timezone non-interactively
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime

RUN apt-get update && \
    apt-get install -y git-lfs python3.9 python3-pip curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda

SHELL ["/bin/bash", "-c"]

RUN curl https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -o /root/anaconda.sh && bash /root/anaconda.sh -b -p /root/miniconda

#RUN eval "$(/root/miniconda/bin/conda shell.bash hook)" && conda init bash



RUN eval "$(/root/miniconda/bin/conda shell.bash hook)" && conda init bash && \
    conda create --name replit python=3.9 -y && \
    conda activate replit && \
    conda install pytorch cpuonly -c pytorch

# Add Miniconda to PATH
ENV PATH /root/miniconda/bin:$PATH

# Install PyTorch, torchvision, torchaudio, and transformers
#RUN pip install torch torchvision torchaudio transformers flask
#RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/replit/replit-code-v1-3b /app/replit-code-v1-3b



# Install CUDA
#RUN apt-key del 7fa2af80 && \
#    curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvidia-cfg1-430_525.105.17-0ubuntu1_amd64.deb && \
#    dpkg -i libnvidia-cfg1-430_525.105.17-0ubuntu1_amd64.deb


# Try to fix error "head: cannot open '/etc/ssl/certs/java/cacerts' for reading: No such file or directory"
RUN rm -rf /etc/ssl/certs/java && mkdir /etc/ssl/certs/java

# Cache apt packages in Docker cache, and install CUDA
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked
#    apt-get update && apt-get install -y cuda-11-8



RUN pip install --upgrade pip setuptools wheel
#RUN pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
#RUN pip install torch==2.0+cpu
#RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
#RUN pip install torch torchvision torchaudio
RUN pip install einops
RUN pip install sentencepiece
RUN pip install transformers==4.29.2
RUN pip install --upgrade pip setuptools wheel
RUN pip install flash-attn==0.2.8
RUN pip install triton==2.0.0.dev20221202
RUN pip install accelerate
RUN pip install deepspeed
RUN apt-get install build-essential
RUN apt install git -y
RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
RUN (echo; echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"') >> /home/$USER/.bashrc
RUN eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
ENV PATH="/home/linuxbrew/.linuxbrew/bin:${PATH}"
RUN brew doctor
RUN brew install mpich
#RUN pip install deepspeed-mii
RUN conda install mpi4py
RUN apt-get clean
#RUN sudo apt purge nvidia*
#RUN sudo apt-get update
#RUN sudo apt-get install --reinstall software-properties-common
RUN apt update
RUN apt install wget
#RUN wget https://us.download.nvidia.com/XFree86/Linux-x86_64/450.80.02/NVIDIA-Linux-x86_64-450.80.02.run
#RUN sudo sh NVIDIA-Linux-x86_64-450.80.02.run
#RUN apk add wget
#RUN apt-get install wget
#RUN sudo apt-get install --reinstall software-properties-common
#RUN sudo apt install software-properties-common
RUN apt-get install sed
RUN pip install tensor_parallel
RUN pip install https://github.com/BlackSamorez/tensor_parallel/archive/main.zip
#RUN sudo apt-add-repository ‘deb https://dl.winehq.org/wine-builds/ubuntu/ bionic main’
#RUN sudo add-apt-repository ppa:graphics-drivers/ppa
#RUN sudo apt-get update
#RUN curl -O https://us.download.nvidia.com/XFree86/Linux-x86_64/450.80.02/NVIDIA-Linux-x86_64-450.80.02.run
#RUN sudo apt install nvidia-driver-450 -s
#RUN wget https://us.download.nvidia.com/XFree86/Linux-x86_64/450.80.02/NVIDIA-Linux-x86_64-450.80.02.run

#RUN wget https://us.download.nvidia.com/XFree86/Linux-x86_64/450.80.02/NVIDIA-Linux-x86_64-450.80.02.run
#RUN chmod +x NVIDIA-Linux-x86_64-450.80.02.run
#RUN sudo systemctl status display-manager
#RUN sudo systemctl stop your-display-manager
#RUN sudo systemctl start your-display-manager
#RUN sudo rmmod nvidia-uvm
#RUN apt-get install get
#ENV BASE_URL=https://us.download.nvidia.com/tesla
#ENV DRIVER_VERSION=450.80.02

#RUN curl -fSsl -O $BASE_URL/$DRIVER_VERSION/NVIDIA-Linux-x86_64-$DRIVER_VERSION.run
#RUN sudo sh NVIDIA-Linux-x86_64-$DRIVER_VERSION.run -s
#RUN sudo service lightdm stop
#RUN sudo ./NVIDIA-Linux-x86_64-450.80.02.run -s

#RUN sudo apt-get install nvidia-450=450.80.02-0ubuntu1
#RUN sed -i 's/world_size = comm.Get_size()\.size()/global_world_size = 2/' /root/miniconda/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py
#RUN sed -i 's/group_world_size = len(ranks)/group_world_size = 1/' /root/miniconda/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py
#RUN sed -i 's/world_size = comm\.Get_size()/world_size = 2/g' /root/miniconda/lib/python3.10/site-packages/deepspeed/comm/comm.py
# Set the CUDA_HOME environment variable
ENV CUDA_HOME /usr/local/cuda

RUN pip install flask termcolor

# Run setup_cuda.py
RUN apt-get update && apt-get install ninja-build

ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
RUN cd /app/replit-code-v1-3b

RUN pip install python-dotenv kserve



COPY . /app
#COPY replit_inference_server.py /app/replit-code-v1-3b/replit_inference_server.py
#COPY rep_cp.py /app/replit-code-v1-3b/rep_cp.py
COPY replit_inference_cp.py /app/replit-code-v1-3b/replit_inference_cp.py
WORKDIR /app

ENV NCCL_SHM_DISABLE=1
ENV PYTHONUNBUFFERED=1
ENV WORLD_SIZE=2
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/replit-code-v1-3b"
ENV NCCL_DEBUG=TRACE
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:8192

CMD ["sleep", "99999999"]
#CMD ["python3", "-m", "replit-code-v1-3b.replit_inference_server", "/mnt/pvc/replit-code-v1"]
#CMD ["deepspeed", "--num_gpus", "4", "/app/replit-code-v1-3b/replit_inference_cp.py"]
#CMD ["python3", "/app/replit-code-v1-3b/rep_cp.py"]
#CMD ["python3", "-m", "replit-code-v1-3b.rep_cp"]
