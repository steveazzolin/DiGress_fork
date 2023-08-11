# Folder to evaluate generated graphs

## GraphRNN

Refer to evaluate_graphrnn.py. 

To evaluate a single batch: ' python evaluate_graphrnn.py --dataset_name sbm --path ../../GraphRNN_fork/graphs/GraphRNN_RNN_sbm_4_128_pred_2900_1.dat'

To evaluate all runs to find the best epoch: 'python evaluate_graphrnn.py --dataset_name sbm --path ../../GraphRNN_fork/graphs/ --find_best=True'

To evaluate for a given epoch all bigger graphs: 'python evaluate_graphrnn.py --dataset_name ego --bigger=1 --epoch 1600'

In the latter case, the reference large datasets are assumed to be in ../../AHK/dataset


## DiGress

Refer to evaluate.py
Arguments are similar as to 'evaluate_graphrnn.py', apart from small differences.