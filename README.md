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


### Prerequisites
All requirements listed in the 'requirements.txt'-file, simply run the following commands:

```
sudo apt-get install python3.7
sudo apt-get install python3-pip
sudo apt-get update
sudo apt-get install python3-venv
git clone https://github.com/Sandbergo/autonomous-vehicle-detector
cd autonomous-vehicle-detector
python3 -m venv env
source env/bin/activate
cd SSD
pip3 install -r requirements.txt
```

### File Structure

The hierarchy should look like this:

    .
    ├── papers                              
    │     └── ...
    ├── SSD                     
    │     ├── data_description.txt
    │     └── train.csv
    |
    ├── .gitignore
    ├── LICENSE
    ├── README.md
    └── requirements.txt


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
