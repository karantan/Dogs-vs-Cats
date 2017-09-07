
Dogs vs. Cats Redux: Kernels Edition
====================================

The purpose of this project is to demonstrate how training CNN model is
preformed. Once we have trained the model we can then use it.


Prerequisites
-------------

You need to have ``conda`` installed on your system.


Installation
------------

Install Dogs-vs-Cats by running:

```bash
git clone https://github.com/karantan/Dogs-vs-Cats
conda env create -f environment.yml
```

After this we can activate ``Dogs-vs-Cats`` environment:

```bash
$ source activate Dogs-vs-Cats
(Dogs-vs-Cats)$ 
```

You will probably also need to change the ``.keras/keras.json`` file so that
it is using ``theano`` and not ``tensorflow``:

```bash
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "image_dim_ordering": "th",
    "backend": "theano"
}
```

And create ``~.theanorc`` config file with the following content:

```bash

[global]
floatX=float32
device=gpu
[mode]=FAST_RUN

[nvcc]
fastmath=True

[cuda]
root=/usr/local/cuda
```



Get the images
--------------

First you need to get the ``data`` for the model to train.

```bash
(Dogs-vs-Cats)$ mkdir data && cd data
(Dogs-vs-Cats)$ wget http://files.fast.ai/data/dogscats.zip
(Dogs-vs-Cats)$ unzip dogscats.zip
```

Get pretrained vgg16 model
--------------------------

Next you need to get the vgg16 model that was already pre-trained based on the
imagenet dogs and cats images.

```bash
(Dogs-vs-Cats)$ mkdir models && cd models
(Dogs-vs-Cats)$ wget http://files.fast.ai/models/vgg16.h5
(Dogs-vs-Cats)$ wget http://files.fast.ai/models/imagenet_class_index.json
```


Train the model
---------------

To train the model you need to run the ``src.train`` script:


```bash
(Dogs-vs-Cats)$ python -m src.train
```

TODO: Implement ``click``.


Run the prediction
------------------

To run a prediction based on the trained model you can use the ``src.predict``
script.

TODO: Implement ``click``.


Install Theano on AWS
---------------------

1) update the default packages
sudo apt-get update
sudo apt-get -y dist-upgrade

2) create a new screen named theano (or look at using Tmux instead) - optional
screen -S "theano"

3) install all the dependencies
sudo apt-get install -y gcc g++ gfortran build-essential git wget linux-image-generic libopenblas-dev python-dev python-pip python-nose python-numpy python-scipy

4) install the bleeding-edge version of Theano
sudo pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

5) grab the latest (7.0) cuda toolkit.
sudo wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb

6) depackage Cuda
sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb

7) add the package and install the cuda driver (this takes a while)
sudo apt-get update
sudo apt-get install -y cuda

8) update the path to include cuda nvcc and ld_library_path
echo -e "\nexport PATH=/usr/local/cuda/bin:$PATH\n\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64" >> .bashrc

9) reboot the system for cuda to load
sudo reboot

Wait a little bit for the reboot and then ssh back into the instance.

10) install included samples and test cuda
cuda-install-samples-7.0.sh ~/
cd NVIDIA\_CUDA-7.0\_Samples/1\_Utilities/deviceQuery
make
./deviceQuery

Make sure it shows that the GPU exists.

11) set up the theano config file to use gpu by default
echo -e "\n[global]\nfloatX=float32\ndevice=gpu\n[mode]=FAST_RUN\n\n[nvcc]\nfastmath=True\n\n[cuda]\nroot=/usr/local/cuda" >> ~/.theanorc


Ref: http://markus.com/install-theano-on-aws/


Further Reading
---------------

TODO

- https://github.com/datitran/Dogs-vs-Cats


Contribute
----------

- Issue Tracker: github.com/karantan/Dogs-vs-Cats/issues
- Source Code: github.com/karantan/Dogs-vs-Cats

Support
-------

If you are having issues, please let us know.

License
-------

MIT License

Copyright (c) 2017 Gasper Vozel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
