#!/usr/bin/env bash
apt-get install sudo
sudo apt-get update 
sudo apt-get upgrade -y 
sudo apt-get install tmux vim zip locales python3-pip zsh git --fix-missing -y  
# sudo apt-get install vim csh flex gfortran libgfortran3 g++ \
                    #  cmake xorg-dev patch zlib1g-dev libbz2-dev \
                    #  libboost-all-dev openssh-server libcairo2 \
                    #  libcairo2-dev libeigen3-dev lsb-core \
                    #  lsb-base net-tools network-manager \
                    #  git-core git-gui git-doc xclip gdebi-core --fix-missing -y 

sudo apt-get install libglib2.0-0 -y 
sudo apt-get install libsm6 libxrender1 libfontconfig1 -y 

export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8" 
sudo dpkg-reconfigure locales 
# 149 3  
sudo pip3 install virtualenv tensorflow-gpu torch numpy scikit-learn scipy nltk gensim pandas keras gdown opencv-python  
virtualenv -p python3 vTA   
source ~/vTA/bin/activate
pip install virtualenv tensorflow-gpu torch numpy scikit-learn scipy nltk gensim pandas keras gdown opencv-python torchvision jeditdistance

alias vTA=~/vTA/bin/python
alias vTA_activate=~/vTA/bin/activate
# source ~/.bashrc

chsh -s /bin/zsh
