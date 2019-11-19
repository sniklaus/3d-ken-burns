# 3d-ken-burns
This is a reference implementation of 3D Ken Burns Effect from a Single Image [1] using PyTorch. Given a single input image, it animates this still image with a virtual camera scan and zoom subject to motion parallax. Should you be making use of our work, please cite our paper [1].

<a href="https://arxiv.org/abs/1909.05483" rel="Paper"><img src="http://content.sniklaus.com/kenburns/paper.jpg" alt="Paper" width="100%"></a>

## setup
To download the pre-trained models, run `bash download.bash`.

Several functions are implemented in CUDA using CuPy, which is why CuPy is a required dependency. It can be installed using `pip install cupy` or alternatively using one of the provided binary packages as outlined in the CuPy repository. Please also make sure to have the `CUDA_HOME` environment variable configured.

In order to generate the video results, please also make sure to have `pip install moviepy` installed.

## usage
To run it on an image and generate the 3D Ken Burns effect fully automatically, use the following command.

```
python autozoom.py --in ./images/doublestrike.jpg --out ./autozoom.mp4
```

To start the interface that allows you to manually adjust the camera path, use the following command. You can then navigate to `http://localhost:8080/` and load an image using the button on the bottom right corner. Please be patient when loading an image and saving the result, there is a bit of background processing going on.

```
python interface.py
```

To run the depth estimation to obtain the raw depth estimate, use the following command. Please note that this script does not perform the depth adjustment, I will add it to the script at a later time should people end up being interested in it.

```
python depthestim.py --in ./images/doublestrike.jpg --out ./depthestim.npy
```

To benchmark the depth estimation, run `python benchmark.py`. You can use it to easily verify that the provided implementation runs as expected.

## video
<a href="http://content.sniklaus.com/kenburns/video.mp4" rel="Video"><img src="http://content.sniklaus.com/kenburns/video.jpg" alt="Video" width="100%"></a>

## license
This is a project by Adobe Research. It is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) and may only be used for non-commercial purposes. Please see the LICENSE file for more information.

## references
```
[1]  @article{Niklaus_TOG_2019,
         author = {Simon Niklaus and Long Mai and Jimei Yang and Feng Liu},
         title = {3D Ken Burns Effect from a Single Image},
         journal = {ACM Transactions on Graphics},
         volume = {38},
         number = {6},
         pages = {184:1--184:15},
         year = {2019}
     }
```

## acknowledgment
The video above uses materials under a Creative Common license or with the owner's permission, as detailed at the end.