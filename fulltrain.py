# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import dataset
import numpy as np
import time
import multi_click_models as clk
import sharedmem
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("output_path", type=str,
                    help="Path to file for pretrained model.")
parser.add_argument("--loss", type=str,
                    default='relevant_rank',
                    help="Loss used for optimization.")
parser.add_argument("--dataset", type=str,
                    default="Webscope_C14_Set1",
                    help="Name of dataset to sample from.")
parser.add_argument("--dataset_info_path", type=str,
                    default="datasets_info.txt",
                    help="Path to dataset info file.")
parser.add_argument("--num_proc", type=int,
                    default=1,
                    help="Number of process to start by multi-threading.")
parser.add_argument("--sigma", type=float,
                    default=1,
                    help="Sigma to use for upper bound.")
parser.add_argument("--cutoff", type=float,
                    default=5,
                    help="Top-k ranking cutoff.")
args = parser.parse_args()

sigma = args.sigma
loss_name = args.loss
cutoff = args.cutoff

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
    if loss_name in ['monotonic', 'lambdaloss@k', 'lambdaloss-full', 'lambdaloss-truncated']:
      cur_filter = np.logical_and(label_filter, np.less(inv_ranking, cutoff))
      result -= 100*np.sum((1./np.log2(inv_ranking+2.))[cur_filter])
    else:
      result += np.sum(inv_ranking[label_filter])
    denom += np.sum(label_filter)

  return result/data_split.num_queries(), result/denom

def optimize(data,
             learning_rate,
             trial_epochs=5,
             max_epochs=100,
             epsilon_thres=0.0001,
             sigma=1.0):
  train_data = data.train
  vali_data = data.validation

  # print '----', starting_learning_rate, learning_rate_decay, '----'

  best_model = np.zeros(train_data.datafold.num_features)
  best_loss = np.inf
  pivot_loss = np.inf
  model = np.zeros(train_data.datafold.num_features)

  start_time = time.time()

  epoch_i = 0
  stop_epoch = trial_epochs
  while epoch_i < min(stop_epoch, max_epochs):
    q_permutation = np.random.permutation(train_data.num_queries())
    for qid in q_permutation:
      rel_docs = np.greater(train_data.query_labels(qid), 2).astype(np.float64)
      if not np.any(rel_docs):
        continue
      q_docs = train_data.query_feat(qid)
      q_range = np.arange(q_docs.shape[0])

      q_scores = np.dot(q_docs, model)
      score_diff = q_scores[:, None] - q_scores[None, :]

      if loss_name in ['relevant_rank', 'monotonic']:
        less_mask = np.less(score_diff, 1.).astype(np.float)

        activation_gradient = -less_mask*rel_docs[:, None]
        if loss_name == 'monotonic':
          up_rank = np.sum(np.maximum(1 - score_diff, 0), axis=1)
          dcg_weights = 1./(np.log2(up_rank+1.)**2*np.log(2)*(up_rank+1))
          activation_gradient *= dcg_weights[:, None]
        activation_gradient[q_range, q_range] -= np.sum(activation_gradient, axis=1)

      elif loss_name in ['lambdaloss@k', 'lambdaloss-full', 'lambdaloss-truncated']:

        q_inv = clk.rank_and_invert(q_scores)[1]

        rel_diff = rel_docs[:, None] - rel_docs[None, :]
        rel_mask = np.less_equal(rel_diff, 0.)

        if loss_name == 'lambdaloss-truncated':
          rnk_vec = np.less(q_inv, cutoff)
          rnk_mask = np.logical_or(rnk_vec[:, None],
                                    rnk_vec[None, :])

          rel_mask = np.logical_or(np.logical_not(rnk_mask), rel_mask)

        rank_diff = np.abs(q_inv[:, None] - q_inv[None, :])
        rank_diff[rel_mask] = 1.

        disc_upp = 1. / np.log2(rank_diff+1.)
        disc_low = 1. / np.log2(rank_diff+2.)
        if loss_name == 'lambdaloss@k':
          disc_upp[np.greater(rank_diff, cutoff)] = 0.
          disc_low[np.greater(rank_diff, cutoff-1)] = 0.

        pair_w = disc_upp - disc_low
        pair_w *= np.abs(rel_diff)
        pair_w[rel_mask] = 0.
        
        score_diff[rel_mask] = 0.
        safe_diff = np.minimum(-score_diff, 500)
        act = 1./(1 + np.exp(safe_diff))
        act[rel_mask] = 0.
        safe_exp = pair_w - 1.
        safe_exp[rel_mask] = 0.

        log2_grad = 1./(act**pair_w*np.log(2))
        power_grad = pair_w*(act)**safe_exp
        sig_grad = act*(1-act)

        activation_gradient = -log2_grad*power_grad*sig_grad

      np.fill_diagonal(activation_gradient,
                       np.diag(activation_gradient)
                       - np.sum(activation_gradient, axis=1))

      per_doc = np.sum(activation_gradient, axis=0)

      model += learning_rate*np.sum(per_doc[:, None]*q_docs, axis=0)

    epoch_i += 1
    _, cur_loss = calc_true_loss(model, vali_data)
    print(epoch_i, '%0.05f' % cur_loss)
    if cur_loss < pivot_loss:
      best_model = model
      best_loss = cur_loss
      if pivot_loss - cur_loss > epsilon_thres:
        pivot_loss = cur_loss
        stop_epoch = epoch_i + trial_epochs

  # print epoch_i, starting_learning_rate, learning_rate_decay, '--- %0.05f' % best_loss

  result = {
      'model': best_model,
      'true_loss': best_loss,
      'epoch': epoch_i - trial_epochs,
      'total_time_spend': time.time()-start_time,
      'time_per_epoch': (time.time()-start_time)/float(epoch_i),
      'learning_rate': learning_rate,
      'sigma': sigma,
      'trial_epochs': trial_epochs,
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


def multi_optimize(args):
  (lr,  sigma) = args
  return optimize(data,
                  learning_rate=lr,
                  trial_epochs=20,
                  max_epochs=500,
                  sigma=sigma)

arg_list = [(lr, 1.) for lr in [10**-x for x in range(7)]]

if args.num_proc > 1:
  pool = Pool(processes=args.num_proc)
  results = pool.map(multi_optimize, arg_list)
else:
  results = [multi_optimize(x) for x in arg_list]

best_result = results[0]
for r in results:
  if r['true_loss'] < best_result['true_loss']:
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