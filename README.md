# RNN Controller
RNN controller code used in paper ["Recurrent Neural Networks for Stochastic Control in Real-Time Bidding"](https://doi.org/10.1145/3292500.3330749) in The 25th ACM SIGKDD Conference ([KDD'19](https://www.kdd.org/kdd2019/)).

## Usage

Directory sample_data contains an example of the data file format to use for training on large datasets.

One can run the following commands and run the main in data.py to ensure correct setup:
```bash
mkdir -p /tmp/noisy_controller_demo_small/2018-09-13/
cp ./sample_data/20180913_1028526 /tmp/noisy_controller_demo_small/2018-09-13/20180913_1028526
```
Alternatively, one can use toy datasets from toy_volume_curves.py as input_fn to the TF estimator provided by experiment.py
