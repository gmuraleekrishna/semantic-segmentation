# Segmentation
A project to semantically segment images with SegNet https://arxiv.org/pdf/1511.00561.pdf

Use KITTI semantic segmentation dataset and create following folder structure

    |_kitty/
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
    $ tensorboard --logdir=graphs/
