# ===================================================================
# CVCWS-UST-PEDESTRIAN-PREDICTION
# -------------------------------------------------------------------
# Dockerfile to create Caffe Image
#
# @author	Technical Support Unit <ust@cvc.uab.es>
#
#           Centro de Vision por Computador
#           Edifici O - Universitat Autonoma de Barcelona.
#           08193 Cerdanyola del Valles, Barcelona, (SPAIN).
#           Tel. +(34) 93.581.18.28
#           Fax. +(34) 93.581.16.70
#
# @version  1.0.0 (7 de Abril de 2021)
# ===================================================================

FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04
LABEL maintainer="ust@cvc.uab.es"

USER root

ARG PYTHON_VERSION=3.6
ARG OPENCV_VERSION=4.5.1

RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    build-essential \
    git \
    zlib1g-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

#The wget is needed to get python 3.6.8 and the others are required to latter on install numpy with pip3.
RUN apt update \
 && apt install wget \
 && apt install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev --no-install-recommends -y

#install python 3.6.8
RUN cd /opt \
    && wget https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tgz \
    && tar -xvf Python-3.6.8.tgz \
    && cd Python-3.6.8 \
    && ./configure \
    && make \
    && make install

# install pip
RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 --no-cache-dir install -U numpy==1.19.5

# Install all dependencies for OpenCV
RUN apt-get -y update -qq --fix-missing && \
    apt-get -y install --no-install-recommends \
        unzip \
        cmake \
        ffmpeg \
        libtbb2 \
        gfortran \
        apt-utils \
        pkg-config \
        checkinstall \
        qt5-default \
        libopenblas-base \
        libopenblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        libjpeg8-dev \
        libpng-dev \
        libtiff5-dev \
        libdc1394-22-dev \
        libxine2-dev \
        libv4l-dev \
        libgstreamer1.0 \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-0 \
        libgstreamer-plugins-base1.0-dev \
        libglew-dev \
        libpostproc-dev \
        libeigen3-dev \
        libtbb-dev \
        libsm6 \
        libxext6 \
        libxrender1 \
	libqt5gui5

# Install OpenCV
RUN wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip -O opencv.zip --progress=bar:force:noscroll && \
    unzip -q opencv.zip && \
    mv /opencv-$OPENCV_VERSION /opencv && \
    rm opencv.zip && \
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip -O opencv_contrib.zip --progress=bar:force:noscroll && \
    unzip -q opencv_contrib.zip && \
    mv /opencv_contrib-$OPENCV_VERSION /opencv_contrib && \
    rm opencv_contrib.zip

# Prepare build
RUN mkdir /opencv/build && \
    cd /opencv/build && \
    cmake \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D BUILD_PYTHON_SUPPORT=ON \
      -D BUILD_DOCS=ON \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_TESTS=OFF \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
      -D BUILD_opencv_python3=$( [ ${PYTHON_VERSION%%.*} -ge 3 ] && echo "ON" || echo "OFF" ) \
      -D BUILD_opencv_python2=$( [ ${PYTHON_VERSION%%.*} -lt 3 ] && echo "ON" || echo "OFF" ) \
      -D PYTHON${PYTHON_VERSION%%.*}_EXECUTABLE=$(which python${PYTHON_VERSION}) \
      -D PYTHON_DEFAULT_EXECUTABLE=$(which python${PYTHON_VERSION}) \
      -D BUILD_EXAMPLES=OFF \
      -D WITH_IPP=OFF \
      -D WITH_FFMPEG=ON \
      -D WITH_GSTREAMER=ON \
      -D WITH_V4L=ON \
      -D WITH_LIBV4L=ON \
      -D WITH_TBB=ON \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D WITH_CUDA=ON \
      -D WITH_LAPACK=ON \
      #-D WITH_HPX=ON \
      -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
      # https://kezunlin.me/post/6580691f
      # https://stackoverflow.com/questions/28010399/build-opencv-with-cuda-support
      -D CUDA_ARCH_BIN="5.3 6.1 7.0 7.5" \
      -D CUDA_ARCH_PTX="" \
      -D WITH_CUBLAS=ON \
      -D WITH_NVCUVID=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D ENABLE_PRECOMPILED_HEADERS=OFF \
      ..

# Build, Test and Install
RUN cd /opencv/build && \
    make -j$(nproc) && \
    make install && \
    ldconfig


# Call default command.
#RUN ffmpeg -version && \
#    #ldd `which ffmpeg` && \
#    python3 --version && \
#    python3 -c "import cv2 ; print(cv2.__version__)"

RUN pip3 install wheel

# install tensorflow (1.13.1 to work with cuda10.1 and cudnn 7.5)
RUN pip3 install tensorflow-gpu==1.13.1

# install pandas, scipy
RUN pip3 install pandas==1.1.5
RUN pip3 install scipy==1.5.4
RUN pip3 install matplotlib==3.3.4
RUN pip3 install sklearn==0.0
RUN pip3 install psutil==5.8.0
RUN pip3 install pyyaml==6.0
RUN pip3 install easydict==1.9
RUN pip3 install --upgrade pip
RUN pip3 install norfair[metrics,video]==0.1.8
RUN pip3 install shapely==1.8.0
RUN pip3 install redis==3.3.0
RUN pip3 install ptvsd==4.3.2
RUN apt-get clean

# cleaning
RUN rm -rf /opencv /opencv_contrib /var/lib/apt/lists/*

# clone the repositories
#First install lfs to get the large files.
RUN apt-get update --no-install-recommends -y 
RUN apt-get install curl --no-install-recommends -y 
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs=3.0.2

RUN git clone https://github.com/FEM-IoT/CVC
RUN git clone https://github.com/avBuffer/Yolov5_tf 

RUN cp /CVC/Docker/utils.py /Yolov5_tf/core
RUN mkdir /Yolov5_tf/TensorFlow
RUN cp /CVC/TensorFlow/femiot_crossing_yolov3.pb /Yolov5_tf/TensorFlow
RUN cp /CVC/TensorFlow/yolov3.pb /Yolov5_tf/TensorFlow
RUN cp /CVC/TensorFlow/coco.names /Yolov5_tf/TensorFlow
RUN cp /CVC/TensorFlow/femiot.names /Yolov5_tf/TensorFlow

RUN cp /CVC/femiotUtils.py /Yolov5_tf
RUN cp /CVC/server_detection.py /Yolov5_tf
RUN cp /CVC/test_detection.py /Yolov5_tf
RUN cp /CVC/femiotUtils.py /Yolov5_tf
RUN cp /CVC/docker-entrypoint.sh /Yolov5_tf
RUN cp /CVC/docker-entrypoint-debug.sh /Yolov5_tf

# correct some incorrect paths
RUN sed -i 's/D:\/datasets\/Social\/labels.txt/\/Yolov5_tf\/data\/classes\/coco.names/g' /Yolov5_tf/core/config.py
RUN sed -i 's/D:\/datasets\/Social\/social_train.txt/\/Yolov5_tf\/data\/dataset\/voc_train.txt/g' /Yolov5_tf/core/config.py
RUN sed -i 's/D:\/datasets\/Social\/social_val.txt/\/Yolov5_tf\/data\/dataset\/voc_test.txt/g' /Yolov5_tf/core/config.py

# Workspace
ENV PYTHONPATH "${PYTHONPATH}:/Yolov5_tf"
WORKDIR /Yolov5_tf

