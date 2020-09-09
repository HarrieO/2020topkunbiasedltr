# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import dataset
import numpy as np
import time
import multi_click_models as clk

parser = argparse.ArgumentParser()
parser.add_argument("model_file", type=str,
                    help="Model file.")
parser.add_argument("--dataset", type=str,
                    default="Webscope_C14_Set1",
                    help="Name of dataset to sample from.")
parser.add_argument("--dataset_info_path", type=str,
                    default="datasets_info.txt",
                    help="Path to dataset info file.")
parser.add_argument("--dataset_partition", type=str,
                    default="validation",
                    help="Partition to use (train/validation/test).")
args = parser.parse_args()

data = dataset.get_dataset_from_json_info(
                  args.dataset,
                  args.dataset_info_path,
                  shared_resource = False,
                )

data = data.get_data_folds()[0]

# start = time.time()
data.read_data()
# print 'Time past for reading data: %d seconds' % (time.time() - start)

ranking_model = clk.read_model(args.model_file, data)

if args.dataset_partition == 'train':
  data_split = data.train
elif args.dataset_partition == 'validation':
  data_split = data.validation
elif args.dataset_partition == 'test':
  data_split = data.test

all_docs = data_split.feature_matrix
all_scores = np.dot(all_docs, ranking_model)

def dcg_at_k(sorted_labels, k):
  if k > 0:
    k = min(sorted_labels.shape[0], k)
  else:
    k = sorted_labels.shape[0]
  denom = 1./np.log2(np.arange(k)+2.)
  nom = 2**sorted_labels-1.
  dcg = np.sum(nom[:k]*denom)
  return dcg

def ndcg_at_k(sorted_labels, ideal_labels, k):
  return dcg_at_k(sorted_labels, k) / dcg_at_k(ideal_labels, k)

def evaluate_query(qid):
  s_i, e_i = data_split.doclist_ranges[qid:qid+2]
  n_docs = e_i - s_i
  q_scores = all_scores[s_i:e_i]
  q_labels = data_split.query_labels(qid)

  random_i = np.random.permutation(
               np.arange(q_scores.shape[0])
             )
  q_labels = q_labels[random_i]
  q_scores = q_scores[random_i]

  sort_ind = np.argsort(q_scores)[::-1]
  sorted_labels = q_labels[sort_ind]
  ideal_labels = np.sort(q_labels)[::-1]

  bin_labels = np.greater(sorted_labels, 2)
  bin_ideal_labels = np.greater(ideal_labels, 2)

  rel_i = np.arange(1,len(sorted_labels)+1)[bin_labels]

  total_labels = float(np.sum(bin_labels))
  assert total_labels > 0 or np.any(np.greater(q_labels, 0))
  if total_labels > 0:
    result = {
      'relevant rank': list(rel_i),
      'relevant rank per query': np.sum(rel_i),
      'precision@01': np.sum(bin_labels[:1])/1.,
      'precision@03': np.sum(bin_labels[:3])/3.,
      'precision@05': np.sum(bin_labels[:5])/5.,
      'precision@10': np.sum(bin_labels[:10])/10.,
      'precision@20': np.sum(bin_labels[:20])/20.,
      'recall@01': np.sum(bin_labels[:1])/total_labels,
      'recall@03': np.sum(bin_labels[:3])/total_labels,
      'recall@05': np.sum(bin_labels[:5])/total_labels,
      'recall@10': np.sum(bin_labels[:10])/total_labels,
      'recall@20': np.sum(bin_labels[:20])/total_labels,
      'dcg': dcg_at_k(sorted_labels, 0),
      'dcg@03': dcg_at_k(sorted_labels, 3),
      'dcg@05': dcg_at_k(sorted_labels, 5),
      'dcg@10': dcg_at_k(sorted_labels, 10),
      'dcg@20': dcg_at_k(sorted_labels, 20),
      'ndcg': ndcg_at_k(sorted_labels, ideal_labels, 0),
      'ndcg@03': ndcg_at_k(sorted_labels, ideal_labels, 3),
      'ndcg@05': ndcg_at_k(sorted_labels, ideal_labels, 5),
      'ndcg@10': ndcg_at_k(sorted_labels, ideal_labels, 10),
      'ndcg@20': ndcg_at_k(sorted_labels, ideal_labels, 20),   
      'binarized_dcg': dcg_at_k(bin_labels, 0),
      'binarized_dcg@03': dcg_at_k(bin_labels, 3),
      'binarized_dcg@05': dcg_at_k(bin_labels, 5),
      'binarized_dcg@10': dcg_at_k(bin_labels, 10),
      'binarized_dcg@20': dcg_at_k(bin_labels, 20),
      'binarized_ndcg': ndcg_at_k(bin_labels, bin_ideal_labels, 0),
      # 'binarized_ndcg@03': ndcg_at_k(bin_labels, bin_ideal_labels, 3),
      # 'binarized_ndcg@05': ndcg_at_k(bin_labels, bin_ideal_labels, 5),
      # 'binarized_ndcg@10': ndcg_at_k(bin_labels, bin_ideal_labels, 10),
      # 'binarized_ndcg@20': ndcg_at_k(bin_labels, bin_ideal_labels, 20),
    }
    for i in range(1, 100):
      result['binarized_ndcg@%02d' % i] = ndcg_at_k(bin_labels, bin_ideal_labels, i)
  else:
    result = {
      'dcg': dcg_at_k(sorted_labels, 0),
      'dcg@03': dcg_at_k(sorted_labels, 3),
      'dcg@05': dcg_at_k(sorted_labels, 5),
      'dcg@10': dcg_at_k(sorted_labels, 10),
      'dcg@20': dcg_at_k(sorted_labels, 20),
      'ndcg': ndcg_at_k(sorted_labels, ideal_labels, 0),
      'ndcg@03': ndcg_at_k(sorted_labels, ideal_labels, 3),
      'ndcg@05': ndcg_at_k(sorted_labels, ideal_labels, 5),
      'ndcg@10': ndcg_at_k(sorted_labels, ideal_labels, 10),
      'ndcg@20': ndcg_at_k(sorted_labels, ideal_labels, 20),
    }
  return result

def included(qid):
  return np.any(np.greater(data_split.query_labels(qid), 0))

def add_to_results(results, cur_results):
  for k, v in cur_results.items():
    if not (k in results):
      results[k] = []
    if type(v) == list:
      results[k].extend(v)
    else:
      results[k].append(v)


results = {}
for qid in np.arange(data_split.num_queries()):
  if included(qid):
    add_to_results(results, evaluate_query(qid))

for k in sorted(results.keys()):
  v = results[k]
  mean_v = np.mean(v)
  std_v = np.std(v)
  print('%s: %0.04f (%0.05f)' % (k, mean_v, std_v))

exit()


def parse_line(line):
  line = line[:line.find('#')]
  splitted = line.split()
  label = int(splitted[0])
  qid = int(splitted[1].split(':')[1])
  feat = {}
  for x in splitted[2:]:
    feat_id, value = x.split(':')
    feat[int(feat_id)] = value
    # if int(feat_id) in model:
    #   result += model[int(feat_id)] * float(value)
  return label, qid, feat

def get_scores(features):
  results = np.zeros(len(features))
  for feat_id in model:
    feat_uniq = [float(c_f[feat_id])
                 for c_f in features if feat_id in c_f]
    if feat_uniq:
      feat_values = np.array([float(c_f.get(feat_id, 0.))
                              for c_f in features])

      feat_mask = np.array([not (feat_id in c_f)
                            for c_f in features])


      feat_values -= np.amin(feat_values)
      safe_max = np.amax(feat_values)
      if safe_max > 0:
        feat_values /= safe_max
      # feat_values[feat_mask] = 0.

      results += feat_values * model[feat_id]

  return list(results)

def ndcg_at_k(sorted_labels, ideal_labels, k):
  if k > 0:
    k = min(sorted_labels.shape[0], k)
  else:
    k = sorted_labels.shape[0]
  denom = 1./np.log2(np.arange(k)+2.)
  nom = 2**sorted_labels-1.
  dcg = np.sum(nom[:k]*denom)

  inom = 2**ideal_labels-1.
  idcg = np.sum(inom[:k]*denom)

  if idcg == 0:
    return 0.
  else:
    return dcg/idcg

def dcg_at_k(sorted_labels, k):
  if k > 0:
    k = min(sorted_labels.shape[0], k)
  else:
    k = sorted_labels.shape[0]
  denom = 1./np.log2(np.arange(k)+2.)
  nom = 2**sorted_labels-1.
  dcg = np.sum(nom[:k]*denom)
  return dcg

def ndcg(labels, scores):
  labels = np.array(labels)
  scores = np.array(scores)

  random_i = np.random.permutation(
               np.arange(scores.shape[0])
             )

  labels = labels[random_i]
  scores = scores[random_i]

  sort_i = np.argsort(-scores)
  sorted_labels = labels[sort_i]
  ideal_labels = np.sort(labels)[::-1]

  bin_labels = np.greater(sorted_labels, 2)
  bin_ideal_labels = np.greater(ideal_labels, 2)

  rel_i = np.arange(1,len(labels)+1)[bin_labels]

  total_labels = float(np.sum(bin_labels))
  if total_labels > 0:
    result = {
      'relevant rank': list(rel_i),
      'relevant rank per query': np.sum(rel_i),
      'precision@01': np.sum(bin_labels[:1])/1.,
      'precision@03': np.sum(bin_labels[:3])/3.,
      'precision@05': np.sum(bin_labels[:5])/5.,
      'precision@10': np.sum(bin_labels[:10])/10.,
      'precision@20': np.sum(bin_labels[:20])/20.,
      'recall@01': np.sum(bin_labels[:1])/total_labels,
      'recall@03': np.sum(bin_labels[:3])/total_labels,
      'recall@05': np.sum(bin_labels[:5])/total_labels,
      'recall@10': np.sum(bin_labels[:10])/total_labels,
      'recall@20': np.sum(bin_labels[:20])/total_labels,
      'dcg': dcg_at_k(sorted_labels, 0),
      'dcg@03': dcg_at_k(sorted_labels, 3),
      'dcg@05': dcg_at_k(sorted_labels, 5),
      'dcg@10': dcg_at_k(sorted_labels, 10),
      'dcg@20': dcg_at_k(sorted_labels, 20),
      'ndcg': ndcg_at_k(sorted_labels, ideal_labels, 0),
      'ndcg@03': ndcg_at_k(sorted_labels, ideal_labels, 3),
      'ndcg@05': ndcg_at_k(sorted_labels, ideal_labels, 5),
      'ndcg@10': ndcg_at_k(sorted_labels, ideal_labels, 10),
      'ndcg@20': ndcg_at_k(sorted_labels, ideal_labels, 20),   
      'binarized_dcg': dcg_at_k(bin_labels, 0),
      'binarized_dcg@03': dcg_at_k(bin_labels, 3),
      'binarized_dcg@05': dcg_at_k(bin_labels, 5),
      'binarized_dcg@10': dcg_at_k(bin_labels, 10),
      'binarized_dcg@20': dcg_at_k(bin_labels, 20),
      # 'binarized_ndcg': ndcg_at_k(bin_labels, bin_ideal_labels, 0),
      # 'binarized_ndcg@03': ndcg_at_k(bin_labels, bin_ideal_labels, 3),
      # 'binarized_ndcg@05': ndcg_at_k(bin_labels, bin_ideal_labels, 5),
      # 'binarized_ndcg@10': ndcg_at_k(bin_labels, bin_ideal_labels, 10),
      # 'binarized_ndcg@20': ndcg_at_k(bin_labels, bin_ideal_labels, 20),
    }
    for i in range(1, 100):
      result['binarized_ndcg@%02d' % i] = ndcg_at_k(bin_labels, bin_ideal_labels, i)
  else:
    result = {
      'dcg': dcg_at_k(sorted_labels, 0),
      'dcg@03': dcg_at_k(sorted_labels, 3),
      'dcg@05': dcg_at_k(sorted_labels, 5),
      'dcg@10': dcg_at_k(sorted_labels, 10),
      'dcg@20': dcg_at_k(sorted_labels, 20),
      'ndcg': ndcg_at_k(sorted_labels, ideal_labels, 0),
      'ndcg@03': ndcg_at_k(sorted_labels, ideal_labels, 3),
      'ndcg@05': ndcg_at_k(sorted_labels, ideal_labels, 5),
      'ndcg@10': ndcg_at_k(sorted_labels, ideal_labels, 10),
      'ndcg@20': ndcg_at_k(sorted_labels, ideal_labels, 20),
    }
  return result

def included(labels):
  return np.any(np.greater(labels, 0))

def add_to_results(results, cur_results):
  for k, v in cur_results.items():
    if not (k in results):
      results[k] = []
    if type(v) == list:
      results[k].extend(v)
    else:
      results[k].append(v)

with open(args.subset_file, 'r') as subset_file:
  current_qid = None
  labels = []
  features = []
  results = {}
  for line in subset_file:
    c_label, c_qid, c_feat = parse_line(line)
    if c_qid != current_qid:
      if included(labels):
        scores = get_scores(features)
        c_res = ndcg(labels, scores)
        add_to_results(results, c_res)
      current_qid = c_qid
      labels = []
      features = []
    labels.append(c_label)
    features.append(c_feat)

for k in sorted(results.keys()):
  v = results[k]
  mean_v = np.mean(v)
  std_v = np.std(v)
  print('%s: %0.04f (%0.05f)' % (k, mean_v, std_v))
