# TFN
Official implementation of TFN: "Twin Fuzzy Networks with Interpolation Consistency Regularization for Weakly-supervised Anomaly Detection" on IEEE Transactions on Fuzzy Systems

## Dependencies
MATLAB R2022b
  
## Data Preparation
* Get datasets mentioned in the paper:
  * Download ".zip" file in this [share link](https://drive.google.com/file/d/1xbFcrjaphq4Fha9AX4L5c0Hni4mFlHvI/view?usp=drive_link), and unzip it to the folder "./Datasets"
* Generate your anomaly detection datasets:
  * All datasets mentioned in the paper are generated using the python code of [DevNet](https://github.com/GuansongPang/deviation-network) proposed in the SIGKDD paper [Deep Anomaly Detection with Deviation Networks](https://dl.acm.org/doi/10.1145/3292500.3330871)
  * To run your own datasets with weakly supervision, name your generated train/test datasets in the following format:
    * Train dataset: "{}_weakly_train_{}_{}.mat".format(origin_name, contamination_rate, num_known_anomalies)
    * Test dataset: "{}_weakly_test_{}_{}.mat".format(origin_name, contamination_rate, num_known_anomalies)


## Experiments
* To fetch the results in the paper:
  * Run "main.m".

## Useful Information
* Results:
  * The "log_{n}.txt" files that keep the display output for experiments are stored in "./Logs" directory. 
  * The corresponding tables "log_Result_{n}.csv" that store results for that experiments are stored in "./Logs" directory.
* Parameter Settings (in "main.m" file):
  1. `EXP`: Set the random seed, and the number of runs for each dataset
  2. `datasets`: List all the origin names of the datasets under weak supervision
  3. `WSAD`: Set the hyperparameters for the weak supervision scenarios. `known_outlier` is the number of known anoamlies. `contamination` is the contamination rate.
  4. `PCA`: `pca` determine whether PCA is used. `min_dim` is the minimum dimension for the usage of PCA. `threshold` is the explained variation ratio.  
  5. `PTRAIN`: In `method`, `fixed` uses the pre-set number of rules `fix_rule` for these datasets, and `search` will search for ideal number of rules resulting *non-empty* clusters between `min_fule` and `max_rule`.
  6. `AUG`: 
     * `num_train` is the number of data pairs (hyperparameter $M$ in the paper). 
     * `c_values` is the list of $[C_{a,a}, C_{u,a}, C_{u,u}]$. 
     * `E_test` is the number of train data sampled in the test phase (hyperparameter $E$ in the paper).
  7. `MIXUP`:
     * `type` determine whether ICR is used, chosen between "No" and "ICR".
     * `M` is the number of virtual training pairs in the ICR process.
     * `gamma` is the weight of the ICR loss.
  8. `REGU`: `lambda` is the weight of the $l_2$ regularization term.
  9. `TRAIN`: `cluster` choose the cluster method used in TFN, choosen between "p_fcm" and "k-means".

## Full Paper
The full paper can be found at [this link](https://ieeexplore.ieee.org/document/10552872) (Early Acess).

## Citation
```
@ARTICLE{cao2024twin,
  author={Cao, Zhi and Shi, Ye and Chang, Yu-Cheng and Yao, Xin and Lin, Chin-Teng},
  journal={IEEE Transactions on Fuzzy Systems}, 
  title={Twin Fuzzy Networks With Interpolation Consistency Regularization for Weakly-Supervised Anomaly Detection}, 
  year={2024},
  volume={},
  number={},
  pages={1-12},
  keywords={Anomaly detection;Training;Prototypes;Uncertainty;Interpolation;Optimization;Knowledge engineering;Data uncertainty;fuzzy c-means clustering;interpolation consistency regularization;twin fuzzy networks;weakly-supervised anomaly detection},
  doi={10.1109/TFUZZ.2024.3412435}}

```