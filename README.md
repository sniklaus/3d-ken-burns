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

To run the depth estimation to obtain the raw depth estimate, use the following command. Please note that this script does not perform the depth adjustment, see [#22](https://github.com/sniklaus/3d-ken-burns/issues/22) for information on how to add it.

```
python depthestim.py --in ./images/doublestrike.jpg --out ./depthestim.npy
```

To benchmark the depth estimation, run `python benchmark.py`. You can use it to easily verify that the provided implementation runs as expected.

## colab
If you do not have a suitable environment to run this projects then you could give Colab a try. It allows you to run the project in the cloud, free of charge. There are several people who provide Colab notebooks that should get you started. A few that I am aware of include one from [Arnaldo Gabriel](https://colab.research.google.com/github/agmm/colab-3d-ken-burns/blob/master/automatic-3d-ken-burns.ipynb), one from [Vlad Alex](https://towardsdatascience.com/very-spatial-507aa847179d), and one from [Ahmed Harmouche](https://github.com/wpmed92/3d-ken-burns-colab).

## dataset
This dataset is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) and may only be used for non-commercial purposes. Please see the LICENSE file for more information.

| scene | mode | color | depth | normal |
|:------|:-----|------:|------:|-------:|
| asdf | flying | [3.7 GB](https://www.dropbox.com/s/6m266avg5vlugie/asdf-flying.zip?dl=0) | [1.0 GB](https://www.dropbox.com/s/kw6rqgebm83nnq8/asdf-flying-depth.zip?dl=0) | [2.9 GB](https://www.dropbox.com/s/3nqhg9pa7gcsd8y/asdf-flying-normal.zip?dl=0) |
| asdf | walking | [3.6 GB](https://www.dropbox.com/s/8hhey3aemvrqo79/asdf-walking.zip?dl=0) | [0.9 GB](https://www.dropbox.com/s/dtz3f8v1yin7e3f/asdf-walking-depth.zip?dl=0) | [2.7 GB](https://www.dropbox.com/s/9qnnqpx4s5rlb4i/asdf-walking-normal.zip?dl=0) |
| blank | flying | [3.2 GB](https://www.dropbox.com/s/t9q8l4kdokmym6s/blank-flying.zip?dl=0) | [1.0 GB](https://www.dropbox.com/s/jx7hyotlr0vumcr/blank-flying-depth.zip?dl=0) | [2.8 GB](https://www.dropbox.com/s/ep0t54l6is65jbf/blank-flying-normal.zip?dl=0) |
| blank | walking | [3.0 GB](https://www.dropbox.com/s/3jv40m8uaph9drf/blank-walking.zip?dl=0) | [0.9 GB](https://www.dropbox.com/s/orljuc04fer4byx/blank-walking-depth.zip?dl=0) | [2.7 GB](https://www.dropbox.com/s/qlxmt4mnzn9llyu/blank-walking-normal.zip?dl=0) |
| chill | flying | [5.4 GB](https://www.dropbox.com/s/brkjln2il4t49qt/chill-flying.zip?dl=0) | [1.1 GB](https://www.dropbox.com/s/oobjykpm3p5por2/chill-flying-depth.zip?dl=0) | [10.8 GB](https://www.dropbox.com/s/4tw9p56warmio1z/chill-flying-normal.zip?dl=0) |
| chill | walking | [5.2 GB](https://www.dropbox.com/s/xdasuvksfvi5ryl/chill-walking.zip?dl=0) | [1.0 GB](https://www.dropbox.com/s/g9eneys0fw8r32a/chill-walking-depth.zip?dl=0) | [10.5 GB](https://www.dropbox.com/s/xz8wwkub5ppxzkj/chill-walking-normal.zip?dl=0) |
| city | flying | [0.8 GB](https://www.dropbox.com/s/jiea1n48b2rwzn8/city-flying.zip?dl=0) | [0.2 GB](https://www.dropbox.com/s/6rzwse2r83hgfvr/city-flying-depth.zip?dl=0) | [0.9 GB](https://www.dropbox.com/s/f79abbsvyb0o9ok/city-flying-normal.zip?dl=0) |
| city | walking | [0.7 GB](https://www.dropbox.com/s/kn11pccrt89brdl/city-walking.zip?dl=0) | [0.2 GB](https://www.dropbox.com/s/nm629tzt62kqykn/city-walking-depth.zip?dl=0) | [0.8 GB](https://www.dropbox.com/s/mupge7xdwa0ezrm/city-walking-normal.zip?dl=0) |
| environment | flying | [1.9 GB](https://www.dropbox.com/s/a7wjmnw9aocrghl/environment-flying.zip?dl=0) | [0.5 GB](https://www.dropbox.com/s/uvff3vfel2gr1f0/environment-flying-depth.zip?dl=0) | [3.5 GB](https://www.dropbox.com/s/xiuncwxl4js13gl/environment-flying-normal.zip?dl=0) |
| environment | walking | [1.8 GB](https://www.dropbox.com/s/96a5b71xn9i90oq/environment-walking.zip?dl=0) | [0.5 GB](https://www.dropbox.com/s/sad0qzm8uua5zon/environment-walking-depth.zip?dl=0) | [3.3 GB](https://www.dropbox.com/s/lra5xxtq775vhy3/environment-walking-normal.zip?dl=0) |
| fort | flying | [5.0 GB](https://www.dropbox.com/s/0pyyqh9reblmz7l/fort-flying.zip?dl=0) | [1.1 GB](https://www.dropbox.com/s/d59wev1gru66opq/fort-flying-depth.zip?dl=0) | [9.2 GB](https://www.dropbox.com/s/knj7muonkhsdobs/fort-flying-normal.zip?dl=0) |
| fort | walking | [4.9 GB](https://www.dropbox.com/s/fzqy9uf16ch1cyh/fort-walking.zip?dl=0) | [1.1 GB](https://www.dropbox.com/s/v0njesos4lh8v51/fort-walking-depth.zip?dl=0) | [9.3 GB](https://www.dropbox.com/s/nmn76g460u0ngwk/fort-walking-normal.zip?dl=0) |
| grass | flying | [1.1 GB](https://www.dropbox.com/s/e57jcymau5bn288/grass-flying.zip?dl=0) | [0.2 GB](https://www.dropbox.com/s/p1hh8huck9ykp47/grass-flying-depth.zip?dl=0) | [1.9 GB](https://www.dropbox.com/s/5l3qbwrl671nnh9/grass-flying-normal.zip?dl=0) |
| grass | walking | [1.1 GB](https://www.dropbox.com/s/wcwj5f2wcen0p6z/grass-walking.zip?dl=0) | [0.2 GB](https://www.dropbox.com/s/zrzzgagixdf7i73/grass-walking-depth.zip?dl=0) | [1.6 GB](https://www.dropbox.com/s/yp8rqwrwmagihec/grass-walking-normal.zip?dl=0) |
| ice | flying | [1.2 GB](https://www.dropbox.com/s/0716le6g2rhmavv/ice-flying.zip?dl=0) | [0.2 GB](https://www.dropbox.com/s/au7imdjkwth5jhj/ice-flying-depth.zip?dl=0) | [2.1 GB](https://www.dropbox.com/s/tkx6cq5xcyklydd/ice-flying-normal.zip?dl=0) |
| ice | walking | [1.2 GB](https://www.dropbox.com/s/twul3xu1rlkve08/ice-walking.zip?dl=0) | [0.2 GB](https://www.dropbox.com/s/dxx630g3aik1snj/ice-walking-depth.zip?dl=0) | [2.0 GB](https://www.dropbox.com/s/q6h0ul9h40dk87v/ice-walking-normal.zip?dl=0) |
| knights | flying | [0.8 GB](https://www.dropbox.com/s/9mlggezivyiq71t/knights-flying.zip?dl=0) | [0.2 GB](https://www.dropbox.com/s/60rd8tzan5yhqna/knights-flying-depth.zip?dl=0) | [1.0 GB](https://www.dropbox.com/s/dhohfl4z3uuixc4/knights-flying-normal.zip?dl=0) |
| knights | walking | [0.8 GB](https://www.dropbox.com/s/t1v6fbj4ye902h5/knights-walking.zip?dl=0) | [0.2 GB](https://www.dropbox.com/s/lk923ivhl2ff53o/knights-walking-depth.zip?dl=0) | [0.9 GB](https://www.dropbox.com/s/fmwvfkxc2ugotrg/knights-walking-normal.zip?dl=0) |
| outpost | flying | [4.8 GB](https://www.dropbox.com/s/m432pb03w84o9rg/outpost-flying.zip?dl=0) | [1.1 GB](https://www.dropbox.com/s/axdc8mpsxlk1r2d/outpost-flying-depth.zip?dl=0) | [7.9 GB](https://www.dropbox.com/s/2z2mxfhd29evbwt/outpost-flying-normal.zip?dl=0) |
| outpost | walking | [4.6 GB](https://www.dropbox.com/s/56xrvw248tmxoq1/outpost-walking.zip?dl=0) | [1.0 GB](https://www.dropbox.com/s/smkxzkps3g8l2bz/outpost-walking-depth.zip?dl=0) | [7.4 GB](https://www.dropbox.com/s/fhq95q3t3apj7ia/outpost-walking-normal.zip?dl=0) |
| pirates | flying | [0.8 GB](https://www.dropbox.com/s/okir7sdhdyja9t6/pirates-flying.zip?dl=0) | [0.2 GB](https://www.dropbox.com/s/5pwbq8jloqnxr9j/pirates-flying-depth.zip?dl=0) | [0.8 GB](https://www.dropbox.com/s/9uyfciai86s6bht/pirates-flying-normal.zip?dl=0) |
| pirates | walking | [0.7 GB](https://www.dropbox.com/s/dds5ynttmex2zrs/pirates-walking.zip?dl=0) | [0.2 GB](https://www.dropbox.com/s/y9ymqs3eucxqebl/pirates-walking-depth.zip?dl=0) | [0.8 GB](https://www.dropbox.com/s/mh1s1p836bldvjk/pirates-walking-normal.zip?dl=0) |
| shooter | flying | [0.9 GB](https://www.dropbox.com/s/1kcl6slxv3x7an5/shooter-flying.zip?dl=0) | [0.2 GB](https://www.dropbox.com/s/h1n5l21q4exl1x6/shooter-flying-depth.zip?dl=0) | [1.1 GB](https://www.dropbox.com/s/nka007kwigs1obt/shooter-flying-normal.zip?dl=0) |
| shooter | walking | [0.9 GB](https://www.dropbox.com/s/ecah8s9zq75f8j6/shooter-walking.zip?dl=0) | [0.2 GB](https://www.dropbox.com/s/yyfk50i4slz34ie/shooter-walking-depth.zip?dl=0) | [1.0 GB](https://www.dropbox.com/s/3u9oakosd30jkfm/shooter-walking-normal.zip?dl=0) |
| shops | flying | [0.2 GB](https://www.dropbox.com/s/ur7hwmyalq6n6xc/shops-flying.zip?dl=0) | [0.1 GB](https://www.dropbox.com/s/dzy34ikcb43hqyh/shops-flying-depth.zip?dl=0) | [0.2 GB](https://www.dropbox.com/s/4vgjkfjba14wtib/shops-flying-normal.zip?dl=0) |
| shops | walking | [0.2 GB](https://www.dropbox.com/s/psefz72gtfmjs3r/shops-walking.zip?dl=0) | [0.1 GB](https://www.dropbox.com/s/olsfpc8o8s5zftw/shops-walking-depth.zip?dl=0) | [0.2 GB](https://www.dropbox.com/s/o106mbnrj8eolbd/shops-walking-normal.zip?dl=0) |
| slums | flying | [0.5 GB](https://www.dropbox.com/s/gfgu3wal5ohqi7r/slums-flying.zip?dl=0) | [0.1 GB](https://www.dropbox.com/s/z1zua3ks54wkv5i/slums-flying-depth.zip?dl=0) | [0.8 GB](https://www.dropbox.com/s/qcbnv4k8p4nhpej/slums-flying-normal.zip?dl=0) |
| slums | walking | [0.5 GB](https://www.dropbox.com/s/yhmc2mwuszy0l1w/slums-walking.zip?dl=0) | [0.1 GB](https://www.dropbox.com/s/x0v1x66sb5enqe4/slums-walking-depth.zip?dl=0) | [0.7 GB](https://www.dropbox.com/s/t48tvfbbeaetfxm/slums-walking-normal.zip?dl=0) |
| subway | flying | [0.5 GB](https://www.dropbox.com/s/6hyuitry5o1msqx/subway-flying.zip?dl=0) | [0.1 GB](https://www.dropbox.com/s/6dmmtxbhtebqvtt/subway-flying-depth.zip?dl=0) | [0.9 GB](https://www.dropbox.com/s/sqgmjhwli4e8xtr/subway-flying-normal.zip?dl=0) |
| subway | walking | [0.5 GB](https://www.dropbox.com/s/p7cfbj47kyjhtpp/subway-walking.zip?dl=0) | [0.1 GB](https://www.dropbox.com/s/lavvq71ex5i726p/subway-walking-depth.zip?dl=0) | [0.9 GB](https://www.dropbox.com/s/dpdnwvvfai12mel/subway-walking-normal.zip?dl=0) |
| temple | flying | [1.7 GB](https://www.dropbox.com/s/64xe1sazmh7kt8i/temple-flying.zip?dl=0) | [0.4 GB](https://www.dropbox.com/s/drexa1trx95y5t5/temple-flying-depth.zip?dl=0) | [3.1 GB](https://www.dropbox.com/s/vrs1swirk8ctzfc/temple-flying-normal.zip?dl=0) |
| temple | walking | [1.7 GB](https://www.dropbox.com/s/k1ouf810mxhuzsb/temple-walking.zip?dl=0) | [0.3 GB](https://www.dropbox.com/s/tuv05foltbemw0h/temple-walking-depth.zip?dl=0) | [2.8 GB](https://www.dropbox.com/s/s972kfix8ef2z1f/temple-walking-normal.zip?dl=0) |
| titan | flying | [6.2 GB](https://www.dropbox.com/s/wikecq62f09o30t/titan-flying.zip?dl=0) | [1.1 GB](https://www.dropbox.com/s/lcdc8s739g9rcte/titan-flying-depth.zip?dl=0) | [11.5 GB](https://www.dropbox.com/s/v3e2kmav5w8qrip/titan-flying-normal.zip?dl=0) |
| titan | walking | [6.0 GB](https://www.dropbox.com/s/atmzb30flu10ya0/titan-walking.zip?dl=0) | [1.1 GB](https://www.dropbox.com/s/tkknorne636zdju/titan-walking-depth.zip?dl=0) | [11.3 GB](https://www.dropbox.com/s/lo44zlzhyw3n1od/titan-walking-normal.zip?dl=0) |
| town | flying | [1.7 GB](https://www.dropbox.com/s/onystfbtktf9qxc/town-flying.zip?dl=0) | [0.3 GB](https://www.dropbox.com/s/h2ftawyj1mmdw65/town-flying-depth.zip?dl=0) | [3.0 GB](https://www.dropbox.com/s/idpealjh09dohsr/town-flying-normal.zip?dl=0) |
| town | walking | [1.8 GB](https://www.dropbox.com/s/6kg710y1iru2q6l/town-walking.zip?dl=0) | [0.3 GB](https://www.dropbox.com/s/glgvnvt30a71su1/town-walking-depth.zip?dl=0) | [3.0 GB](https://www.dropbox.com/s/keq11btehlhlia9/town-walking-normal.zip?dl=0) |
| underland | flying | [5.4 GB](https://www.dropbox.com/s/zz9sygjj00kf48j/underland-flying.zip?dl=0) | [1.2 GB](https://www.dropbox.com/s/kk5y5j49sa1n4jw/underland-flying-depth.zip?dl=0) | [12.1 GB](https://www.dropbox.com/s/kwhb9ngzm1fpqi9/underland-flying-normal.zip?dl=0) |
| underland | walking | [5.1 GB](https://www.dropbox.com/s/mv3aoak61w6ryj2/underland-walking.zip?dl=0) | [1.2 GB](https://www.dropbox.com/s/6hu5xqk55xsg0yn/underland-walking-depth.zip?dl=0) | [11.4 GB](https://www.dropbox.com/s/my36okxgodf063x/underland-walking-normal.zip?dl=0) |
| victorian | flying | [0.5 GB](https://www.dropbox.com/s/9f2uw0u6q78vuix/victorian-flying.zip?dl=0) | [0.1 GB](https://www.dropbox.com/s/yawqdgcg26nbn75/victorian-flying-depth.zip?dl=0) | [0.8 GB](https://www.dropbox.com/s/5fbh893ttuo0fdh/victorian-flying-normal.zip?dl=0) |
| victorian | walking | [0.4 GB](https://www.dropbox.com/s/qj77dwx34upenuf/victorian-walking.zip?dl=0) | [0.1 GB](https://www.dropbox.com/s/ljwqcdir31w4s2w/victorian-walking-depth.zip?dl=0) | [0.7 GB](https://www.dropbox.com/s/hl0pgo7m1k6ayi8/victorian-walking-normal.zip?dl=0) |
| village | flying | [1.6 GB](https://www.dropbox.com/s/6wnu4be1combjnh/village-flying.zip?dl=0) | [0.3 GB](https://www.dropbox.com/s/4vhxugaup5p1fva/village-flying-depth.zip?dl=0) | [2.8 GB](https://www.dropbox.com/s/ina5crpiwy92t40/village-flying-normal.zip?dl=0) |
| village | walking | [1.6 GB](https://www.dropbox.com/s/jsv480rgq54h86v/village-walking.zip?dl=0) | [0.3 GB](https://www.dropbox.com/s/cohrawnbrsgu43e/village-walking-depth.zip?dl=0) | [2.7 GB](https://www.dropbox.com/s/nhw6p3s0ranus2s/village-walking-normal.zip?dl=0) |
| warehouse | flying | [0.9 GB](https://www.dropbox.com/s/4nd4urrp40ulmva/warehouse-flying.zip?dl=0) | [0.2 GB](https://www.dropbox.com/s/uuxywfcuonjhp44/warehouse-flying-depth.zip?dl=0) | [1.5 GB](https://www.dropbox.com/s/7ih831hvq2bsw9z/warehouse-flying-normal.zip?dl=0) |
| warehouse | walking | [0.8 GB](https://www.dropbox.com/s/d5gv8yi0sa4brfj/warehouse-walking.zip?dl=0) | [0.2 GB](https://www.dropbox.com/s/n1qjgw1vgkuz4uw/warehouse-walking-depth.zip?dl=0) | [1.4 GB](https://www.dropbox.com/s/1ff8oh9q17tu16o/warehouse-walking-normal.zip?dl=0) |
| western | flying | [0.8 GB](https://www.dropbox.com/s/in3s7ovyw6rt3sg/western-flying.zip?dl=0) | [0.2 GB](https://www.dropbox.com/s/7ljiqeh1elakjyi/western-flying-depth.zip?dl=0) | [0.9 GB](https://www.dropbox.com/s/flzy0sphey8m9pg/western-flying-normal.zip?dl=0) |
| western | walking | [0.7 GB](https://www.dropbox.com/s/7fglrda0tb2qpio/western-walking.zip?dl=0) | [0.2 GB](https://www.dropbox.com/s/mkhkj4kwk6f6yfi/western-walking-depth.zip?dl=0) | [0.8 GB](https://www.dropbox.com/s/v1by71sugmf1iag/western-walking-normal.zip?dl=0) |

Please note that this is an updated version of the dataset that we have used in our paper. So while it has fewer scenes in total, each sample capture now has a varying focal length which should help with generalizability. Furthermore, some examples are either over- or under-exposed and it would be a good idea to remove these outliers.

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