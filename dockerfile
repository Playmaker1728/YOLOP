# 继承PyTorch的cuda镜像
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

LABEL maintainer = "1224425503@qq.com"
LABEL version = "0.2"
LABEL description = "prepare deep learning environment"

# 指定docker镜像中，默认的工作路径是/app
WORKDIR /home/BE

COPY ./training/ /home/BE/

RUN rm /etc/apt/sources.list.d/cuda.list \
	&& rm /etc/apt/sources.list.d/nvidia-ml.list \
	&& apt-get update \
	&& apt-get install -y libgl1 \
	&& conda install libgdal=3.4.1 gdal==3.4.1 tiledb=2.2 \
	&& pip config set global.index-url https://pypi.douban.com/simple/ \
	&& pip install opencv-python-headless==4.6.0.66 \
    	opencv-python==4.6.0.66 \
    	easycython==1.0.7 \
    	cython==0.29.30 \
    	pretrainedmodels==0.7.4 \
		efficientnet-pytorch==0.6.3 \
    	segmentation-models-pytorch==0.2.1 \
    	scipy>=1.4.1 \
    	path==16.4.0 \
    	scikit-image==0.15.0 \
    	scikit-learn==0.21.3 \
        pytorch_toolbelt==0.5.0  \
        prefetch_generator \
		yacs \
		tensorboardX


