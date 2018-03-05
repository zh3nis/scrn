#!/usr/bin/env bash

# clean up any previous installation
curl -s https://raw.githubusercontent.com/torch/ezinstall/master/clean-old.sh | bash
rm -rf ~/torch

# install torch dependencies
curl -sk https://raw.githubusercontent.com/torch/ezinstall/7f2fe2f7b9bbc24da1a3544df73b08e8d4fbda43/install-deps | bash -e

# install torch itself
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch
git checkout e0c565120622f99ef6e1ca7fccca66cfe2da34fc
git submodule update
./install.sh


# install Folly, fbthrift, thpp and fblualib
echo
echo This script will install fblualib and all its dependencies.
echo It has been tested on Ubuntu 13.10 and Ubuntu 14.04, Linux x86_64.
echo

set -e
set -x

if [[ $(arch) != 'x86_64' ]]; then
    echo "x86_64 required" >&2
    exit 1
fi

issue=$(cat /etc/issue)
extra_packages=libiberty-dev

dir=$(mktemp --tmpdir -d fblualib-build.XXXXXX)

echo Working in $dir
echo
cd $dir

echo Installing required packages
echo
sudo apt-get install -y \
    git \
    curl \
    wget \
    g++ \
    automake \
    autoconf \
    autoconf-archive \
    libtool \
    libboost-all-dev \
    libevent-dev \
    libdouble-conversion-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    liblz4-dev \
    liblzma-dev \
    libsnappy-dev \
    make \
    zlib1g-dev \
    binutils-dev \
    libjemalloc-dev \
    $extra_packages \
    flex \
    bison \
    libkrb5-dev \
    libsasl2-dev \
    libnuma-dev \
    pkg-config \
    libssl-dev \
    libedit-dev \
    libmatio-dev \
    libpython-dev \
    libpython3-dev \
    python-numpy

echo
echo Cloning repositories
echo
git clone https://github.com/facebook/folly.git
git clone https://github.com/facebook/fbthrift.git
git clone https://github.com/facebook/thpp
git clone https://github.com/soumith/fblualib

echo
echo Building folly
echo

cd $dir/folly
git checkout 2184167af16ed892c4a1cd76866c7dde6f1483c8
cd folly
autoreconf -ivf
./configure
make
sudo make install
sudo ldconfig

echo
echo Building fbthrift
echo

cd $dir/fbthrift
git checkout 2374d889d0f821e8e798178e6b2bee2d0dc8432c
cd thrift
autoreconf -ivf
./configure
make
sudo make install

echo
echo 'Installing TH++'
echo

cd $dir/thpp
git checkout 9e1799d8ce5356ccfc5a17da52fc36bd7d6b426d
cd thpp
wget https://github.com/google/googletest/archive/release-1.7.0.zip
unzip release-1.7.0.zip
mv googletest-release-1.7.0 gtest-1.7.0
set -o pipefail

if [[ ! -r ./Tensor.h ]]; then
  echo "Please run from the thpp subdirectory." >&2
  exit 1
fi

# Build in a separate directory
mkdir -p build
cd build

# Configure
cmake ..

# Make
make

# Run tests
ctest

# Install
sudo make install


echo
echo 'Installing FBLuaLib'
echo

cd $dir/fblualib
git checkout eb050eaeb33af4aed89db078c90ac5f7d4204691
cd fblualib
./build.sh

echo
echo 'All done!'
echo

cd ~/Downloads
mkdir tmp
cd tmp

git clone https://github.com/torch/nn 
cd nn 
git checkout e9fba08c97cdb6da84ef10aac5dc9910d4f9523b
luarocks make rocks/nn-scm-1.rockspec
cd ..

git clone https://github.com/facebook/fbtorch.git 
cd fbtorch 
git checkout 75568061ee9231892d6f0d47c1b556e23a18dbd6
luarocks make rocks/fbtorch-scm-1.rockspec
cd ..

git clone https://github.com/facebook/fbnn.git 
cd fbnn 
git checkout 5dc9bb691436a7687026f4f39b2eea1c0b523ae8
luarocks make rocks/fbnn-scm-1.rockspec
cd ..

git clone https://github.com/facebook/fbcunn.git 
cd fbcunn 
git checkout 34df6926779902952a0d6877fe62857f2f4b258c
luarocks make rocks/fbcunn-scm-1.rockspec

luarocks install nngraph

