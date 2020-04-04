# :blue_book: Benchmarking

This document seeks to provide a benchmarking procedure for comparing the results of different detector models. It can be used as a guide to more easily gauge a model's relative performance.

**All benchmark training should be run 10 000 iterations on the waymo dataset. After having been trained in the waymo dataset, the model should then be run 10 000 iterations on the tdt4265 dataset.**  

---

### Create or modify a network model

Start with building your network model, either by importing a [pre-trained model](https://pytorch.org/docs/stable/torchvision/models.html) or by making your own backbone from scratch. 

Some files that may be a good starting point are:
* `SSD/ssd/modeling/backbone.py`
* `SSD/ssd/solver/build.py`
* `SSD/data/transforms/__init__.py`

You should also remember to change configuration parameters to match your network, e.g. by changing the learning rate, batch size or other parameters. 

> | :exclamation:  Note   |
> |-----------------------|
> When making a new, or modifying an existing, `.yaml` configuration file in `SSD/configs`, be aware that any option added here will overwrite the default configuration file in `SSD/ssd/config/defaults.py`.


### Benchmark configurations

**A preliminary suggestion for benchmarking different models is as follows:**

1. Add a new model backbone file `SSD/ssd/modeling/backbone/my_model.py`
2. Make optional changes to other network components such as the solver/optimizer or the dataloader.
3. Tune the learning rate and batch size of your model, or tweak other solver parameters (ex. by adding momentum).
4. Remove potenital old logs and saved models from your local `SSD/outputs/waymo_bench` and `SSD/outputs/tdt4265_bench` folders if they already exist. 

 After model building, remember to swap the name parameter in the provided `train_waymo_bench.yaml` and `train_tdt4265_bench.yaml` configuration files:

```yaml
MODEL:
    # ...
    BACKBONE:
        # ...
        NAME: 'my_model'
        # ...
    # ..
```

Next, train your new model:

```python
python3 train.py configs/train_waymo_bench.yaml    # Pre-trained == False

python3 train.py configs/train_tdt4265_bench.yaml  # Pre-trained == True
```

### Save results and store models

After each of the training runs, make note of the final reported mAP, and log it in the Drive file `Models/Model log`. Save your final model `.pt`-file in the Drive folder `Models`.

