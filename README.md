# Semantic Segmentation
A project to semantically segment images with SegNet-like architecture

The reference paper [1] can be found at https://arxiv.org/pdf/1511.00561.pdf

SegNet-like because the Maxpooling indices are not shared to decoder.

Use KITTI semantic segmentation dataset and create following folder structure

    |_kitti/
    |    |__train/
    |    |   |_images/
    |    |   |_labels/
    |    |__test/
    |        |_images/
    |        |_labels/
    |_main.py
    |_helper.py

## Usage
    $ pip install -r requirements.txt
    $ python main.py
    $ tensorboard --logdir=graph/
    
References:

 [1]   V. Badrinarayanan, A. Kendall, and R. Cipolla, "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation",  *arXiv:1511.005eprint61*, 2015
