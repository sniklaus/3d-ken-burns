# 3d-ken-burns
This is a reference implementation of 3D Ken Burns Effect from a Single Image [1] using PyTorch. Given a single input image, it animates this still image with a virtual camera scan and zoom subject to motion parallax. Should you be making use of our work, please cite our paper [1].

<a href="https://arxiv.org/abs/1909.05483" rel="Paper"><img src="http://content.sniklaus.com/kenburns/paper.jpg" alt="Paper" width="100%"></a>

## setup
Several functions are implemented in CUDA using CuPy, which is why CuPy is a required dependency. It can be installed using `pip install cupy` or alternatively using one of the provided [binary packages](https://docs.cupy.dev/en/stable/install.html#installing-cupy) as outlined in the CuPy repository. Please also make sure to have the `CUDA_HOME` environment variable configured.

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

To run the depth estimation to obtain the raw depth estimate, use the following command. Please note that this script does not perform the depth adjustment, see [#22](https://github.com/sniklaus/3d-ken-burns/issues/22) for information on how to add it.

```
python depthestim.py --in ./images/doublestrike.jpg --out ./depthestim.npy
```

To benchmark the depth estimation, run `python benchmark-ibims.py` or `python benchmark-nyu.py`. You can use it to easily verify that the provided implementation runs as expected.

## colab
If you do not have a suitable environment to run this projects then you could give Colab a try. It allows you to run the project in the cloud, free of charge. There are several people who provide Colab notebooks that should get you started. A few that I am aware of include one from [Arnaldo Gabriel](https://colab.research.google.com/github/agmm/colab-3d-ken-burns/blob/master/automatic-3d-ken-burns.ipynb), one from [Vlad Alex](https://towardsdatascience.com/very-spatial-507aa847179d), and one from [Ahmed Harmouche](https://github.com/wpmed92/3d-ken-burns-colab).

## dataset
This dataset is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) and may only be used for non-commercial purposes. Please see the LICENSE file for more information.

| scene | mode | color | depth | normal |
|:------|:-----|------:|------:|-------:|
| asdf | flying | [3.7 GB](http://content.sniklaus.com/kenburns/asdf-flying.zip) | [1.0 GB](http://content.sniklaus.com/kenburns/asdf-flying-depth.zip) | [2.9 GB](http://content.sniklaus.com/kenburns/asdf-flying-normal.zip) |
| asdf | walking | [3.6 GB](http://content.sniklaus.com/kenburns/asdf-walking.zip) | [0.9 GB](http://content.sniklaus.com/kenburns/asdf-walking-depth.zip) | [2.7 GB](http://content.sniklaus.com/kenburns/asdf-walking-normal.zip) |
| blank | flying | [3.2 GB](http://content.sniklaus.com/kenburns/blank-flying.zip) | [1.0 GB](http://content.sniklaus.com/kenburns/blank-flying-depth.zip) | [2.8 GB](http://content.sniklaus.com/kenburns/blank-flying-normal.zip) |
| blank | walking | [3.0 GB](http://content.sniklaus.com/kenburns/blank-walking.zip) | [0.9 GB](http://content.sniklaus.com/kenburns/blank-walking-depth.zip) | [2.7 GB](http://content.sniklaus.com/kenburns/blank-walking-normal.zip) |
| chill | flying | [5.4 GB](http://content.sniklaus.com/kenburns/chill-flying.zip) | [1.1 GB](http://content.sniklaus.com/kenburns/chill-flying-depth.zip) | [10.8 GB](http://content.sniklaus.com/kenburns/chill-flying-normal.zip) |
| chill | walking | [5.2 GB](http://content.sniklaus.com/kenburns/chill-walking.zip) | [1.0 GB](http://content.sniklaus.com/kenburns/chill-walking-depth.zip) | [10.5 GB](http://content.sniklaus.com/kenburns/chill-walking-normal.zip) |
| city | flying | [0.8 GB](http://content.sniklaus.com/kenburns/city-flying.zip) | [0.2 GB](http://content.sniklaus.com/kenburns/city-flying-depth.zip) | [0.9 GB](http://content.sniklaus.com/kenburns/city-flying-normal.zip) |
| city | walking | [0.7 GB](http://content.sniklaus.com/kenburns/city-walking.zip) | [0.2 GB](http://content.sniklaus.com/kenburns/city-walking-depth.zip) | [0.8 GB](http://content.sniklaus.com/kenburns/city-walking-normal.zip) |
| environment | flying | [1.9 GB](http://content.sniklaus.com/kenburns/environment-flying.zip) | [0.5 GB](http://content.sniklaus.com/kenburns/environment-flying-depth.zip) | [3.5 GB](http://content.sniklaus.com/kenburns/environment-flying-normal.zip) |
| environment | walking | [1.8 GB](http://content.sniklaus.com/kenburns/environment-walking.zip) | [0.5 GB](http://content.sniklaus.com/kenburns/environment-walking-depth.zip) | [3.3 GB](http://content.sniklaus.com/kenburns/environment-walking-normal.zip) |
| fort | flying | [5.0 GB](http://content.sniklaus.com/kenburns/fort-flying.zip) | [1.1 GB](http://content.sniklaus.com/kenburns/fort-flying-depth.zip) | [9.2 GB](http://content.sniklaus.com/kenburns/fort-flying-normal.zip) |
| fort | walking | [4.9 GB](http://content.sniklaus.com/kenburns/fort-walking.zip) | [1.1 GB](http://content.sniklaus.com/kenburns/fort-walking-depth.zip) | [9.3 GB](http://content.sniklaus.com/kenburns/fort-walking-normal.zip) |
| grass | flying | [1.1 GB](http://content.sniklaus.com/kenburns/grass-flying.zip) | [0.2 GB](http://content.sniklaus.com/kenburns/grass-flying-depth.zip) | [1.9 GB](http://content.sniklaus.com/kenburns/grass-flying-normal.zip) |
| grass | walking | [1.1 GB](http://content.sniklaus.com/kenburns/grass-walking.zip) | [0.2 GB](http://content.sniklaus.com/kenburns/grass-walking-depth.zip) | [1.6 GB](http://content.sniklaus.com/kenburns/grass-walking-normal.zip) |
| ice | flying | [1.2 GB](http://content.sniklaus.com/kenburns/ice-flying.zip) | [0.2 GB](http://content.sniklaus.com/kenburns/ice-flying-depth.zip) | [2.1 GB](http://content.sniklaus.com/kenburns/ice-flying-normal.zip) |
| ice | walking | [1.2 GB](http://content.sniklaus.com/kenburns/ice-walking.zip) | [0.2 GB](http://content.sniklaus.com/kenburns/ice-walking-depth.zip) | [2.0 GB](http://content.sniklaus.com/kenburns/ice-walking-normal.zip) |
| knights | flying | [0.8 GB](http://content.sniklaus.com/kenburns/knights-flying.zip) | [0.2 GB](http://content.sniklaus.com/kenburns/knights-flying-depth.zip) | [1.0 GB](http://content.sniklaus.com/kenburns/knights-flying-normal.zip) |
| knights | walking | [0.8 GB](http://content.sniklaus.com/kenburns/knights-walking.zip) | [0.2 GB](http://content.sniklaus.com/kenburns/knights-walking-depth.zip) | [0.9 GB](http://content.sniklaus.com/kenburns/knights-walking-normal.zip) |
| outpost | flying | [4.8 GB](http://content.sniklaus.com/kenburns/outpost-flying.zip) | [1.1 GB](http://content.sniklaus.com/kenburns/outpost-flying-depth.zip) | [7.9 GB](http://content.sniklaus.com/kenburns/outpost-flying-normal.zip) |
| outpost | walking | [4.6 GB](http://content.sniklaus.com/kenburns/outpost-walking.zip) | [1.0 GB](http://content.sniklaus.com/kenburns/outpost-walking-depth.zip) | [7.4 GB](http://content.sniklaus.com/kenburns/outpost-walking-normal.zip) |
| pirates | flying | [0.8 GB](http://content.sniklaus.com/kenburns/pirates-flying.zip) | [0.2 GB](http://content.sniklaus.com/kenburns/pirates-flying-depth.zip) | [0.8 GB](http://content.sniklaus.com/kenburns/pirates-flying-normal.zip) |
| pirates | walking | [0.7 GB](http://content.sniklaus.com/kenburns/pirates-walking.zip) | [0.2 GB](http://content.sniklaus.com/kenburns/pirates-walking-depth.zip) | [0.8 GB](http://content.sniklaus.com/kenburns/pirates-walking-normal.zip) |
| shooter | flying | [0.9 GB](http://content.sniklaus.com/kenburns/shooter-flying.zip) | [0.2 GB](http://content.sniklaus.com/kenburns/shooter-flying-depth.zip) | [1.1 GB](http://content.sniklaus.com/kenburns/shooter-flying-normal.zip) |
| shooter | walking | [0.9 GB](http://content.sniklaus.com/kenburns/shooter-walking.zip) | [0.2 GB](http://content.sniklaus.com/kenburns/shooter-walking-depth.zip) | [1.0 GB](http://content.sniklaus.com/kenburns/shooter-walking-normal.zip) |
| shops | flying | [0.2 GB](http://content.sniklaus.com/kenburns/shops-flying.zip) | [0.1 GB](http://content.sniklaus.com/kenburns/shops-flying-depth.zip) | [0.2 GB](http://content.sniklaus.com/kenburns/shops-flying-normal.zip) |
| shops | walking | [0.2 GB](http://content.sniklaus.com/kenburns/shops-walking.zip) | [0.1 GB](http://content.sniklaus.com/kenburns/shops-walking-depth.zip) | [0.2 GB](http://content.sniklaus.com/kenburns/shops-walking-normal.zip) |
| slums | flying | [0.5 GB](http://content.sniklaus.com/kenburns/slums-flying.zip) | [0.1 GB](http://content.sniklaus.com/kenburns/slums-flying-depth.zip) | [0.8 GB](http://content.sniklaus.com/kenburns/slums-flying-normal.zip) |
| slums | walking | [0.5 GB](http://content.sniklaus.com/kenburns/slums-walking.zip) | [0.1 GB](http://content.sniklaus.com/kenburns/slums-walking-depth.zip) | [0.7 GB](http://content.sniklaus.com/kenburns/slums-walking-normal.zip) |
| subway | flying | [0.5 GB](http://content.sniklaus.com/kenburns/subway-flying.zip) | [0.1 GB](http://content.sniklaus.com/kenburns/subway-flying-depth.zip) | [0.9 GB](http://content.sniklaus.com/kenburns/subway-flying-normal.zip) |
| subway | walking | [0.5 GB](http://content.sniklaus.com/kenburns/subway-walking.zip) | [0.1 GB](http://content.sniklaus.com/kenburns/subway-walking-depth.zip) | [0.9 GB](http://content.sniklaus.com/kenburns/subway-walking-normal.zip) |
| temple | flying | [1.7 GB](http://content.sniklaus.com/kenburns/temple-flying.zip) | [0.4 GB](http://content.sniklaus.com/kenburns/temple-flying-depth.zip) | [3.1 GB](http://content.sniklaus.com/kenburns/temple-flying-normal.zip) |
| temple | walking | [1.7 GB](http://content.sniklaus.com/kenburns/temple-walking.zip) | [0.3 GB](http://content.sniklaus.com/kenburns/temple-walking-depth.zip) | [2.8 GB](http://content.sniklaus.com/kenburns/temple-walking-normal.zip) |
| titan | flying | [6.2 GB](http://content.sniklaus.com/kenburns/titan-flying.zip) | [1.1 GB](http://content.sniklaus.com/kenburns/titan-flying-depth.zip) | [11.5 GB](http://content.sniklaus.com/kenburns/titan-flying-normal.zip) |
| titan | walking | [6.0 GB](http://content.sniklaus.com/kenburns/titan-walking.zip) | [1.1 GB](http://content.sniklaus.com/kenburns/titan-walking-depth.zip) | [11.3 GB](http://content.sniklaus.com/kenburns/titan-walking-normal.zip) |
| town | flying | [1.7 GB](http://content.sniklaus.com/kenburns/town-flying.zip) | [0.3 GB](http://content.sniklaus.com/kenburns/town-flying-depth.zip) | [3.0 GB](http://content.sniklaus.com/kenburns/town-flying-normal.zip) |
| town | walking | [1.8 GB](http://content.sniklaus.com/kenburns/town-walking.zip) | [0.3 GB](http://content.sniklaus.com/kenburns/town-walking-depth.zip) | [3.0 GB](http://content.sniklaus.com/kenburns/town-walking-normal.zip) |
| underland | flying | [5.4 GB](http://content.sniklaus.com/kenburns/underland-flying.zip) | [1.2 GB](http://content.sniklaus.com/kenburns/underland-flying-depth.zip) | [12.1 GB](http://content.sniklaus.com/kenburns/underland-flying-normal.zip) |
| underland | walking | [5.1 GB](http://content.sniklaus.com/kenburns/underland-walking.zip) | [1.2 GB](http://content.sniklaus.com/kenburns/underland-walking-depth.zip) | [11.4 GB](http://content.sniklaus.com/kenburns/underland-walking-normal.zip) |
| victorian | flying | [0.5 GB](http://content.sniklaus.com/kenburns/victorian-flying.zip) | [0.1 GB](http://content.sniklaus.com/kenburns/victorian-flying-depth.zip) | [0.8 GB](http://content.sniklaus.com/kenburns/victorian-flying-normal.zip) |
| victorian | walking | [0.4 GB](http://content.sniklaus.com/kenburns/victorian-walking.zip) | [0.1 GB](http://content.sniklaus.com/kenburns/victorian-walking-depth.zip) | [0.7 GB](http://content.sniklaus.com/kenburns/victorian-walking-normal.zip) |
| village | flying | [1.6 GB](http://content.sniklaus.com/kenburns/village-flying.zip) | [0.3 GB](http://content.sniklaus.com/kenburns/village-flying-depth.zip) | [2.8 GB](http://content.sniklaus.com/kenburns/village-flying-normal.zip) |
| village | walking | [1.6 GB](http://content.sniklaus.com/kenburns/village-walking.zip) | [0.3 GB](http://content.sniklaus.com/kenburns/village-walking-depth.zip) | [2.7 GB](http://content.sniklaus.com/kenburns/village-walking-normal.zip) |
| warehouse | flying | [0.9 GB](http://content.sniklaus.com/kenburns/warehouse-flying.zip) | [0.2 GB](http://content.sniklaus.com/kenburns/warehouse-flying-depth.zip) | [1.5 GB](http://content.sniklaus.com/kenburns/warehouse-flying-normal.zip) |
| warehouse | walking | [0.8 GB](http://content.sniklaus.com/kenburns/warehouse-walking.zip) | [0.2 GB](http://content.sniklaus.com/kenburns/warehouse-walking-depth.zip) | [1.4 GB](http://content.sniklaus.com/kenburns/warehouse-walking-normal.zip) |
| western | flying | [0.8 GB](http://content.sniklaus.com/kenburns/western-flying.zip) | [0.2 GB](http://content.sniklaus.com/kenburns/western-flying-depth.zip) | [0.9 GB](http://content.sniklaus.com/kenburns/western-flying-normal.zip) |
| western | walking | [0.7 GB](http://content.sniklaus.com/kenburns/western-walking.zip) | [0.2 GB](http://content.sniklaus.com/kenburns/western-walking-depth.zip) | [0.8 GB](http://content.sniklaus.com/kenburns/western-walking-normal.zip) |

Please note that this is an updated version of the dataset that we have used in our paper. So while it has fewer scenes in total, each sample capture now has a varying focal length which should help with generalizability. Furthermore, some examples are either over- or under-exposed and it would be a good idea to remove these outliers. Please see [#37](https://github.com/sniklaus/3d-ken-burns/issues/37), [#39](https://github.com/sniklaus/3d-ken-burns/issues/39), and [#40](https://github.com/sniklaus/3d-ken-burns/issues/40) for supplementary discussions.

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