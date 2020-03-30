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
git clone https://github.com/Sandbergo/autonomous-vehicle-detector
cd autonomous-vehicle-detector/SSD
pip3 install --user -r requirements.txt
python3 setup_waymo.py
python3 train.py configs/train_waymo.yaml
python3 update_tdt4265_dataset.py
python3 train.py configs/train_tdt4265.yaml
python3 submit_results.py configs/train_tdt4265.yaml
```

Remember to refresh file explorer if working on VS Code server with SSH - remote extension

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

    .
    ├── papers                              
    │     └── ...
    ├── SSD                     
    │     ├── configs
    │     └── requirements.txt
    |
    ├── .gitignore
    ├── LICENSE
    └── README.md
    

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
