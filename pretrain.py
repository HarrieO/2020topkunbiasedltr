# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import dataset
import numpy as np
import time
import stochasticranking as stcrnk
import multi_click_models as clk
import sharedmem
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("output_path", type=str,
                    help="Path to file for pretrained model.")
parser.add_argument("--dataset", type=str,
                    default="Webscope_C14_Set1",
                    help="Name of dataset to sample from.")
parser.add_argument("--dataset_info_path", type=str,
                    default="datasets_info.txt",
                    help="Path to dataset info file.")
parser.add_argument("--percentage", type=float,
                    default=0.01,
                    help="Percentage of data to train on.")
parser.add_argument("--num_proc", type=int,
                    default=1,
                    help="Number of process to start by multi-threading.")
args = parser.parse_args()
percentage = args.percentage

data = dataset.get_dataset_from_json_info(
                  args.dataset,
                  args.dataset_info_path,
                  shared_resource = args.num_proc > 1,
                )

data = data.get_data_folds()[0]

start = time.time()
data.read_data()
print('Time past for reading data: %d seconds' % (time.time() - start))


def calc_true_loss(ranking_model, data_split):
  all_docs = data_split.feature_matrix
  all_scores = np.dot(all_docs, ranking_model)
  result = 0.
  denom = 0.
  for qid in range(data_split.num_queries()):
    s_i, e_i = data_split.doclist_ranges[qid:qid+2]
    q_scores = all_scores[s_i:e_i]
    q_labels = data_split.query_labels(qid)

    label_filter = np.greater(q_labels, 2)
    inv_ranking = clk.rank_and_invert(q_scores)[1]
    result += np.sum(inv_ranking[label_filter])
    denom += np.sum(label_filter)

  return result/data_split.num_queries(), result/denom

def calc_sub_loss(ranking_model, data_split, query_selection):
  all_docs = data_split.feature_matrix
  all_scores = np.dot(all_docs, ranking_model)
  result = 0.
  denom = 0.
  for qid in query_selection:
    s_i, e_i = data_split.doclist_ranges[qid:qid+2]
    q_scores = all_scores[s_i:e_i]
    q_labels = data_split.query_labels(qid)

    label_filter = np.greater(q_labels, 2)
    inv_ranking = clk.rank_and_invert(q_scores)[1]
    result += np.sum(inv_ranking[label_filter])
    denom += np.sum(label_filter)

  return result/query_selection.shape[0], result/denom

def optimize(data,
             train_queries,
             validation_queries,
             learning_rate,
             trial_epochs=3,
             max_epochs=200,
             epsilon_thres=0.001,
             learning_rate_decay=0.9999):
  starting_learning_rate = learning_rate

  train_data = data.train
  vali_data = data.validation

  best_model = np.zeros(train_data.datafold.num_features)
  best_loss = np.inf
  pivot_loss = np.inf
  model = np.zeros(train_data.datafold.num_features)

  start_time = time.time()

  epoch_i = 0
  stop_epoch = trial_epochs
  while epoch_i < min(stop_epoch, max_epochs):
    q_permutation = np.random.permutation(train_queries)
    for qid in q_permutation:
      rel_docs = np.greater(train_data.query_labels(qid), 2)
      if not np.any(rel_docs):
        continue
      d_permutation = np.random.permutation(np.where(rel_docs)[0])
      q_docs = train_data.query_feat(qid)
      q_scores = np.dot(q_docs, model)

      for d_i in d_permutation:
        clicked_score = q_scores[d_i]
        click_doc = q_docs[d_i, :]

        less_mask = np.less_equal(clicked_score, q_scores+1.)
        less_docs = q_docs[less_mask, :]

        gradient = click_doc*less_docs.shape[0]-np.sum(less_docs, axis=0)
        model += learning_rate*gradient

        learning_rate *= learning_rate_decay

    epoch_i += 1
    _, cur_loss = calc_sub_loss(model, vali_data, validation_queries)
    # print(epoch_i, '%0.05f' % cur_loss, cur_loss - best_loss)
    if cur_loss < pivot_loss:
      best_model = model
      best_loss = cur_loss
      if pivot_loss - cur_loss > epsilon_thres:
        pivot_loss = cur_loss
        stop_epoch = epoch_i + trial_epochs

  _, true_loss = calc_true_loss(best_model,
                                vali_data)

  # print(epoch_i, starting_learning_rate, '%0.06f' % learning_rate_decay, '%0.05f' % best_loss, '%0.05f' % true_loss)

  result = {
      'model': best_model,
      'estimated_loss': best_loss,
      'true_loss': true_loss,
      'epoch': epoch_i - trial_epochs,
      'total_time_spend': time.time()-start_time,
      'time_per_epoch': (time.time()-start_time)/float(epoch_i),
      'learning_rate': starting_learning_rate,
      'learning_rate_decay': learning_rate_decay,
      'trial_epochs': trial_epochs,
      'percentage_of_data': percentage,
    }
  return result

def _make_shared(numpy_matrix):
    """
    Avoids the copying of Read-Only shared memory.
    """
    if numpy_matrix is None:
      return None
    else:
      shared = sharedmem.empty(numpy_matrix.shape,
                               dtype=numpy_matrix.dtype)
      shared[:] = numpy_matrix[:]
      return shared

train_q = np.random.permutation(data.train.num_queries())
train_q = train_q[:int(data.train.num_queries()*percentage) + 1]
vali_q = np.random.permutation(data.validation.num_queries())
vali_q = vali_q[:int(data.validation.num_queries()*percentage) + 1]

def multi_optimize(args):
  lr, decay = args
  return optimize(data,
                  train_q,
                  vali_q,
                  learning_rate=lr,
                  trial_epochs=10,
                  learning_rate_decay=decay)


arg_list = [(lr, 1.) for lr in np.geomspace(10**0., 10**-9., 20)]

if args.num_proc > 1:
  pool = Pool(processes=args.num_proc)
  results = pool.map(multi_optimize, arg_list)
else:
  results = [multi_optimize(x) for x in arg_list]

best_result = results[0]
for r in results:
  if r['estimated_loss'] < best_result['estimated_loss']:
    best_result = r

def _doc_feat_str(doc_feat):
  doc_str = ""
  for f_i, f_v in enumerate(doc_feat):
    if f_v == 1.:
      doc_str += ' %d:1' % data.feature_map[f_i]
    elif f_v != 0.:
      doc_str += ' %d:%f' % (data.feature_map[f_i], f_v)
  return doc_str

output = ''
output += '--Simulation Arguments--\n'
output += '--Model Found--\n'
for k in sorted(best_result.keys()):
  if k != 'model':
    output += '%s: %s\n' % (k, best_result[k])
output += '1 %s\n' % _doc_feat_str(best_result['model'])

with open(args.output_path, 'w') as f:
  f.write(output)

print(output)