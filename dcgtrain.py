# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import dataset
import sharedmem
import multi_click_models as clk
import numpy as np
import os
import time
from multiprocessing import Pool
import utils.arp_ips as ips

parser = argparse.ArgumentParser()
parser.add_argument("loss", type=str,
                    help="Loss to optimize.",
                    default='monotonic')
parser.add_argument("model_file", type=str,
                    help="Model file output from pretrained model.")
parser.add_argument("output_path", type=str,
                    help="Path to output model.")
parser.add_argument("--click_model", type=str,
                    help="Name of click model to use.",
                    default='binarized')
parser.add_argument("--dataset", type=str,
                    default="Webscope_C14_Set1",
                    help="Name of dataset to sample from.")
parser.add_argument("--dataset_info_path", type=str,
                    default="datasets_info.txt",
                    help="Path to dataset info file.")
parser.add_argument("--eta", type=float,
                    default=1.0,
                    help="Eta parameter for observance probabilities.")
parser.add_argument("--cutoff", type=int,
                    default=5,
                    help="Item selection cutoff in ranking.")
parser.add_argument("--num_clicks", type=int,
                    default=10000000,
                    help="Number of clicks to generate.")
parser.add_argument("--scale", type=float,
                    default=5.0,
                    help="Scaling the linear scorer.")
parser.add_argument("--num_proc", type=int,
                    default=1,
                    help="Number of processes to use.")

args = parser.parse_args()

prop_model = 'replacelast_policyaware_clipped'
cutoff = args.cutoff
eta = args.eta
num_proc = args.num_proc
assert num_proc >= 0, 'Invalid number of processes: %d' % num_proc

num_clicks = args.num_clicks
c = np.log(num_clicks)/np.log(10)
if args.dataset == 'Webscope_C14_Set1':
  a = -3.
  b = 32.
elif args.dataset == 'MSLR-WEB30k':
  # MSLR
  a = -57
  b = 86.
elif args.dataset == 'istella':
  # Istella
  a = 30.
  b = 0.

clip_thres = (c-4.)*(a) + (c-4.)**2.*b + 1.
clip_thres = max(clip_thres, 1.)
if c < 4:
  clip_thres = 1.

data = dataset.get_dataset_from_json_info(
                  args.dataset,
                  args.dataset_info_path,
                  shared_resource = num_proc > 1,
                )

data = data.get_data_folds()[0]

start = time.time()
data.read_data()
print('Time past for reading data: %d seconds' % (time.time() - start))

pretrain_model = clk.read_model(args.model_file, data, args.scale)

num_train_clicks = int(num_clicks)
num_validation_clicks = int(num_train_clicks*0.15)

start = time.time()
train_clicks = clk.generate_squashed_clicks(
                    prop_model,
                    data.train,
                    pretrain_model,
                    args.click_model,
                    num_train_clicks,
                    cutoff,
                    eta,
                    clip_thres)
print('Time past for generating train clicks: %d seconds' % (time.time() - start))
start = time.time()
validation_clicks = clk.generate_squashed_clicks(
                    prop_model,
                    data.validation,
                    pretrain_model,
                    args.click_model,
                    num_validation_clicks,
                    cutoff,
                    eta,
                    0)
print('Time past for generating validation clicks: %d seconds' % (time.time() - start))


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

if args.num_proc > 1:
  train_clicks['average_weights'] = _make_shared(train_clicks['average_weights'])
  train_clicks['clicks_per_doc'] = _make_shared(train_clicks['clicks_per_doc'])
  train_clicks['queries'] = _make_shared(train_clicks['queries'])
  validation_clicks['average_weights'] = _make_shared(validation_clicks['average_weights'])
  validation_clicks['clicks_per_doc'] = _make_shared(validation_clicks['clicks_per_doc'])
  validation_clicks['queries'] = _make_shared(validation_clicks['queries'])


epsilon_thres=0.000001

base_loss = args.loss.replace('_cutoff', '').replace('_nocutoff', '')

# lr_range = np.geomspace(10**-4., 10**-11., 30)
click_order = np.log(args.num_clicks)/np.log(10)
if args.dataset == 'Webscope_C14_Set1':
  if args.loss in ['log_monotonic', 'monotonic', ]:
    lr_range = [100.]
  elif args.loss in ['lambdaloss@k', 'lambdaloss-full']:
    lr_range = [50.]
  elif args.loss in ['lambdaloss-truncated']:
    lr_range = [10.]
  else:
    assert False
  tries = 30+int(10*click_order)
elif args.dataset == 'MSLR-WEB30k':
  # MSLR
  if args.loss == 'log_monotonic':
    lr_range = [50.]
  elif args.loss == 'monotonic':
    lr_range = [500.]
  elif args.loss in ['lambdaloss-truncated', 'lambdaloss@k', 'lambdaloss-full']:
    lr_range = [50.]
  else:
    assert False

  tries = 30+int(10*click_order)
elif args.dataset == 'istella':
  # Istella
  lr_range = [.01]
  tries = 20+int(2*click_order)

def multi_optimize(m_args):
  lr = m_args
  l_cutoff = cutoff
  if '_nocutoff' in args.loss:
    l_cutoff = 0
  return ips.optimize_dcg(
                  args.loss.replace('_cutoff', '').replace('_nocutoff', ''),
                  data,
                  train_clicks,
                  validation_clicks,
                  learning_rate=lr,
                  learning_rate_decay=1.,
                  trial_epochs=tries,
                  max_epochs=1000,
                  epsilon_thres=epsilon_thres,
                  cutoff=l_cutoff,
                 )

arg_list = lr_range

if args.num_proc > 1:
  pool = Pool(processes=args.num_proc)
  results = pool.map(multi_optimize, arg_list)
else:
  results = [multi_optimize(x) for x in arg_list]

best_result = results[0]
for r in results:
  if r['estimated_loss'] < best_result['estimated_loss']:
    best_result = r

best_result['clipping_threshold'] = clip_thres

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
output += 'propensity model: %s\n' % prop_model
output += 'loss function: %s\n' % args.loss
output += 'click_model model: %s\n' % args.click_model
output += 'number of clicks: %s\n' % args.num_clicks
output += 'eta: %s\n' % args.eta
output += 'cutoff: %s\n' % args.cutoff
output += 'dataset: %s\n' % args.dataset
output += '--Model Found--\n'
for k in sorted(best_result.keys()):
  if k != 'model':
    output += '%s: %s\n' % (k, best_result[k])
output += '1 %s\n' % _doc_feat_str(best_result['model'])
print(output)

with open(args.output_path, 'w') as f:
  f.write(output)