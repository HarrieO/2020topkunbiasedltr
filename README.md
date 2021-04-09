# Policy-Aware Unbiased Learning to Rank for Top-k Rankings
This repository contains the code used for the experiments in "Policy-Aware Unbiased Learning to Rank for Top-k Rankings" published at SIGIR 2020. The paper is available (here)[https://dl.acm.org/doi/abs/10.1145/3397271.3401102], alternatively, a pre-print can be found (here)[https://arxiv.org/abs/2005.09035].

Citation
--------

If you use this code to produce results for your scientific publication, or if you share a copy or fork, please refer to our SIGIR 2020 paper:

```
@inproceedings{Oosterhuis2020Unbiased,
  title={Policy-Aware Unbiased Learning to Rank for Top-k Rankings},
  author={Oosterhuis, Harrie and de Rijke, Maarten},
  booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2020},
  organization={ACM}
}
```

License
-------

The contents of this repository are licensed under the [MIT license](LICENSE). If you modify its contents in any way, please link back to this repository.


Usage
-------

This code makes use of [Python 3](https://www.python.org/) the [numpy](https://numpy.org/) and [sharedmem](https://github.com/rainwoodman/sharedmem) packages, make sure they are installed.

Then a file is required that explains the location and details of the LTR datasets available on the system, for the Yahoo! Webscope, MSLR-Web30k, and Istella datasets an example file is available. Copy the file:
```
cp example_datasets_info.txt datasets_info.txt
```
Open this copy and edit the paths to the folders where the train/test/vali files are placed.

Here are some command-line examples that illustrate how the results in the paper can be replicated.
First create a folder to store the resulting models:
```
mkdir local_output
```
We will start by creating the pre-trained model, trained on 1% of training data:
```
python3 pretrain.py local_output/pretrained_model.txt --dataset_info_path datasets_info.txt  --dataset Webscope_C14_Set1 --num_proc 1
```
To evaluate the resulting model, the following command will print a large number of metrics:
```
python3 evaluate.py local_output/pretrained_model.txt --dataset_info_path datasets_info.txt --dataset Webscope_C14_Set1 --dataset_partition test 
```
Then we can train the full-information model (note that *--num_proc* can be set to the number of processess available for multiprocessing, to speed things up):
```
python3 fulltrain.py local_output/full_information_model.txt --loss lambdaloss-truncated --dataset_info_path datasets_info.txt  --dataset Webscope_C14_Set1 --num_proc 1
```
Again we can evaluate the resulting model, which should perform much better than the pre-trained model:
```
python3 evaluate.py local_output/full_information_model.txt --dataset_info_path datasets_info.txt --dataset Webscope_C14_Set1 --dataset_partition test 
```
To reproduce Figure 1 and 2 you can use *topktrain.py*.
The following command will simulate 10000 clicks using a pretrained model on a top-5 rankings, and subsequently, apply the Policy-Aware Estimator (with replace-last randomization) to optimize the truncated LambdaLoss:
```
python3 topktrain.py replacelast_policyaware  output/Webscope_C14_Set1/supervised/pretrained/pretrained_model.txt local_output/model.txt --loss lambdaloss-truncated --dataset_info_path datasets_info.txt  --dataset Webscope_C14_Set1 --num_proc 1 --num_clicks 10000 --cutoff 5
```
The same setup but optimizing *ARP* only requires a change of the *--loss* flag:
```
python3 topktrain.py replacelast_policyaware  output/Webscope_C14_Set1/supervised/pretrained/pretrained_model.txt local_output/model.txt --loss relevant_rank --dataset_info_path datasets_info.txt  --dataset Webscope_C14_Set1 --num_proc 1 --num_clicks 10000 --cutoff 5
```
Optimizing *ARP* with the Policy-Oblivious Estimator (with replace-last randomization) only requires a change of the first argument:
```
python3 topktrain.py replacelast_oblivious  output/Webscope_C14_Set1/supervised/pretrained/pretrained_model.txt local_output/model.txt --loss relevant_rank --dataset_info_path datasets_info.txt  --dataset Webscope_C14_Set1 --num_proc 1 --num_clicks 10000 --cutoff 5
```
To reproduce Figure 3 you can use *dcgtrain.py*, this code always applies the Policy-Aware Estimator (with replace-last randomization) but allows you to optimize different losses for DCG optimization.
The following optimizes the truncated-lambdaloss:
```
python3 dcgtrain.py lambdaloss-truncated output/Webscope_C14_Set1/supervised/pretrained/pretrained_model.txt local_output/model.txt --dataset_info_path datasets_info.txt  --dataset Webscope_C14_Set1 --num_proc 1 --num_clicks 10000 --cutoff 5
```
Again evaluation can be done with:
```
python3 evaluate.py local_output/model.txt --dataset_info_path datasets_info.txt --dataset Webscope_C14_Set1 --dataset_partition test 
```

The following estimators and randomization options are included:
1. deterministic - no randomization and full-ranking i.e. no cutoff at top-k
2. deterministic_cutoff - no randomization and top-k ranking and the policy-oblivious estimator
3. deterministic_cutoff_rerank - no randomization and top-k ranking and the reranking estimator
4. replacelast_oblivious - replace-last randomization and the policy-oblivious estimator
5. replacelast_oblivious_rerank - replace-last randomization and the reranking estimator
6. replacelast_policyaware - replace-last randomization and the policy-aware estimator

Adding *_clipped* at the end of these options also enables propensity clipping, this is recommended for variance reduction.

The following loss function options are included:
1. monotonic
2. log_monotonic
3. lambdaloss-full
4. lambdaloss@k
5. lambdaloss-truncated
6. relevant_rank

All options are implemented for *dcgtrain.py*, *topktrain.py* only supports *lambdaloss-truncated* and *relevant_rank*.
