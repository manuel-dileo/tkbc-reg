# 
# Temporal Regularisers for Neural Link Predictor
This code reproduces results in Enhancing neural link predictors for temporal knowledge graphs with temporal regularisers, accepted at ESANN 2025.  

This repository is under development and it may contain also additional experiments, datasets, and models not discussed in our manuscript. However, below we provide all the instructions to reproduce the results.

## Installation
Create a conda environment with pytorch and scikit-learn :
```
conda create --name tkbc_env python=3.7
source activate tkbc_env
conda install --file requirements.txt -c pytorch
```

Then install the kbc package to this environment
```
python setup.py install
```

## Datasets

To download the datasets, go to the tkbc/scripts folder and run:
```
chmod +x download_tntcomplex_data.sh
./download_data.sh
```

Once the datasets are downloaded, add them to the package data folder by running :
```
python tkbc/process_icews.py
python tkbc/process_yago.py
```

This will create the files required to compute the filtered metrics.

## Reproducing results

In order to reproduce the best results reported in the paper, run the following commands

```
python tkbc/learner.py --dataset ICEWS14 --model TNTComplEx --rank 2000 --time_reg smooth --time_reg_w 0.01 --time_norm Np --p_norm 4 --emb_reg 0.001

python tkbc/learner.py --dataset ICEWS05-15 --model TNTComplEx --rank 2000 --time_reg smooth --time_reg_w 1 --time_norm Np --p_norm 5 --emb_reg 0.001

python tkbc/learner.py --dataset yago15k --model TNTComplEx --rank 2000 --time_reg smooth --time_reg_w 0.0001 --time_norm Np --p_norm 5 --emb_reg 0.0001 
```

### Reproducing ChronoR results
The original implementation of ChronoR (2021) is not available, we re-implement the solution in our framework. In order to reproduce the best ChronoR results, run the following commands

```
python tkbc/learner.py --dataset ICEWS14 --model ChronoR --rank 2000 --time_reg smooth --time_reg_w 0.0001 --time_norm Lp --p_norm 4 --emb_reg 0.01

python tkbc/learner.py --dataset ICEWS05-15 --model ChronoR --rank 2000 --time_reg smooth --time_reg_w 0.0001 --time_norm Lp --p_norm 3 --emb_reg 0.01

python tkbc/learner.py --dataset yago15k --model ChronoR --rank 2000 --time_reg smooth --time_reg_w 1 --time_norm Lp --p_norm 2 --emb_reg 0.01 
```

### Reproducing TeLM results
The original implementation of TeLM (2021) is [available](https://github.com/soledad921/TeLM). In order to reproduce the results reported in our manuscript, you can follow their instruction and set the rank equal to 1000.

### Use of RNN regularisers
In order to use recurrent architecture as implicit temporal regularisers, run commands like the following:
```
python tkbc/learner.py --dataset ICEWS14 --model RTComplEx --rank 2000 --rnn GRU --rnn_size 500 --emb_reg 0.01
```




## License
tkbc-reg is CC-BY-NC licensed, as found in the LICENSE file.
