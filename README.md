<h1 align="center">Autonomous Vehicle Detector</h1>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]() 
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center">
Single Shot Detector for Autonomus Vehicle Vision. Detects vehicles, pedestrians and signs. Trained on data from Waymo and Trondheim. 
</p>
<br> 

## 📝 Table of Contents
- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Built Using](#built_using)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>
Lets go bois

## 🏁 Getting Started <a name = "getting_started"></a>
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

```
ssh username@clab[00-25].idi.ntnu.no
mk_work_dir
cd ../../../../work/username
rm -rf . // dobbeltsjekk denne kommandoen hehe
git clone https://github.com/Sandbergo/autonomous-vehicle-detection.git
cd autonomous-vehicle-detector/SSD
rm -rf ~/.local/lib/python3.6/site-packages/
rm -rf ~/.local/lib/python2.7/site-packages/
pip3 install --user torch torchvision
pip3 install --user -r requirements.txt
pip3 install --user --upgrade torch torchvision
pip3 install --user tensorflow
pip3 install --upgrade --user pandas
python3 setup_waymo.py
python3 train.py configs/train_waymo.yaml
python3 update_tdt4265_dataset.py
python3 train.py configs/train_tdt4265.yaml
python3 submit_results.py configs/train_tdt4265.yaml
```

Remember to refresh file explorer if working on VS Code server with SSH - remote extension

#### :movie_camera: Video Maker
Download videoes from Google Drive, put in `datasets/videoes/`.

```bash
mkdir outputs/videos // Not tested
snap install ffmpeg
python3 demo_video.py configs/train_tdt4265.yaml datasets/videos/2019-12-05_18-26-20-front_split2.mp4 outputs/videos/output1.mp4
python3 demo_video.py configs/train_tdt4265.yaml datasets/videos/2019-12-06_09-44-38-front_split1.mp4 outputs/videos/output2.mp4
```


### Access Tensorboard
First, on terminal in Cybele:
```
tensorboard --logdir outputs
```
Note the resulting port XXXX.
Then, on a separate terminal running on local computer: 
```
ssh -L 127.0.0.1:6008:127.0.0.1:XXXX username@clab[00-25].idi.ntnu.no
```
Acess on localhost:6008

### File Structure

The hierarchy should look like this:

    ./
    ├── LICENSE
    ├── papers
    │   └── project.pdf
    ├── README.md
    └── SSD
        ├── configs
        │   ├── train_tdt4265.yaml
        │   └── train_waymo.yaml
        ├── demo.ipynb
        ├── demo.py
        ├── demo_video.py
        ├── download_waymo.py
        ├── plot_scalars.ipynb
        ├── README.md
        ├── requirements.txt
        ├── setup_waymo.py
        ├── ssd
        │   ├── config
        │   │   ├── defaults.py
        │   │   └── path_catlog.py
        │   ├── container.py
        │   ├── data
        │   │   ├── build.py
        │   │   ├── datasets
        │   │   │   ├── evaluation
        │   │   │   │   ├── coco
        │   │   │   │   │   └── __init__.py
        │   │   │   │   ├── __init__.py
        │   │   │   │   ├── mnist
        │   │   │   │   │   └── __init__.py
        │   │   │   │   ├── voc
        │   │   │   │   │   ├── eval_detection_voc.py
        │   │   │   │   │   └── __init__.py
        │   │   │   │   └── waymo
        │   │   │   │       └── __init__.py
        │   │   │   ├── __init__.py
        │   │   │   ├── mnist_object_detection
        │   │   │   │   ├── mnist_object_dataset.py
        │   │   │   │   ├── mnist.py
        │   │   │   │   └── visualize_dataset.py
        │   │   │   ├── tdt4265.py
        │   │   │   └── waymo.py
        │   │   ├── samplers.py
        │   │   └── transforms
        │   │       ├── __init__.py
        │   │       ├── target_transform.py
        │   │       └── transforms.py
        │   ├── engine
        │   │   ├── inference.py
        │   │   └── trainer.py
        │   ├── modeling
        │   │   ├── backbone
        │   │   │   ├── basic.py
        │   │   │   └── vgg.py
        │   │   ├── box_head
        │   │   │   ├── box_head.py
        │   │   │   ├── inference.py
        │   │   │   ├── loss.py
        │   │   │   └── prior_box.py
        │   │   └── detector.py
        │   ├── solver
        │   │   ├── build.py
        │   │   └── lr_scheduler.py
        │   ├── torch_utils.py
        │   └── utils
        │       ├── box_utils.py
        │       ├── checkpoint.py
        │       ├── logger.py
        │       ├── metric_logger.py
        │       ├── model_zoo.py
        │       └── nms.py
        ├── submit_results.py
        ├── test.ipynb
        ├── test.py
        ├── train.ipynb
        ├── train.py
        ├── tutorials
        │   ├── annotation_images
        │   │   ├── canvas_completed.png
        │   │   ├── canvas.png
        │   │   ├── canvas_shape_part1.png
        │   │   ├── canvas_shape_part2.png
        │   │   ├── canvas_shape_part3.png
        │   │   ├── create_shape.png
        │   │   ├── login_edit.png
        │   │   ├── task_assignee_edit.png
        │   │   └── tasks_edit.png
        │   ├── annotation_tutorial.md
        │   ├── dataset.md
        │   ├── environment_setup.md
        │   ├── evaluation_tdt4265.md
        │   ├── run.md
        │   └── tensorboard.md
        ├── update_tdt4265_dataset.py
        ├── visualize_dataset.ipynb
        └── visualize_dataset.py

    

## 🎈 Usage <a name="usage"></a>
use & abuse

## ⛏️ Built Using <a name = "built_using"></a>
- [Python 3.7](https://www.python.org/) 
    
    
## ✍️ Authors <a name = "authors"></a>
- Lars Sandberg [@Sandbergo](https://github.com/Sandbergo)
- Sondre Olsen [@sondreo](https://github.com/sondreo)
- Bendik Austnes [@kidneb7](https://github.com/kidneb7)
- Olav Pedersen [@olavpe](https://github.com/olavpe)

## 🎉 Acknowledgements <a name = "acknowledgement"></a>
- Håkon Hukkelås [@hukkelas](https://github.com/hukkelas)
