# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import time
import multi_click_models as clk

def calc_true_loss(ranking_model, data_split):
  all_docs = data_split.feature_matrix
  all_scores = np.dot(all_docs, ranking_model)
  result = 0.
  denom = 0.
  for qid in np.arange(data_split.num_queries()):
    s_i, e_i = data_split.doclist_ranges[qid:qid+2]
    q_scores = all_scores[s_i:e_i]
    q_labels = data_split.query_labels(qid)

    label_filter = np.greater(q_labels, 2)
    inv_ranking = clk.rank_and_invert(q_scores)[1]
    result += np.sum(inv_ranking[label_filter]+1.)
    denom += np.sum(label_filter)

  return result/denom

def estimate_loss(ranking_model,
                  data_split,
                  clicks):
  total_clicks = clicks['num_clicks']
  doc_clicks = clicks['clicks_per_doc'].astype(np.float64)
  inv_prop = clicks['average_weights']*(doc_clicks/float(total_clicks))
  inv_prop /= np.sum(inv_prop)
  all_docs = data_split.feature_matrix
  all_scores = np.dot(all_docs, ranking_model)

  rerank = clicks['rerank']
  if rerank:
    rerank_ranges = clicks['rerank_ranges']
    n_doc_per_q = (data_split.doclist_ranges[1:]
                   - data_split.doclist_ranges[:-1])

    def rerank_get(qid, values):
      s_j, e_j = rerank_ranges[qid:qid+2]
      n = n_doc_per_q[qid]
      return np.reshape(values[s_j:e_j], (n, n))

  result = 0.
  denom = 0.
  for qid in clicks['queries']:
    s_i, e_i = data_split.doclist_ranges[qid:qid+2]
    q_scores = all_scores[s_i:e_i]

    if not rerank:
      q_inv_prop = inv_prop[s_i:e_i]
      inv_ranking = clk.rank_and_invert(q_scores)[1]+1.
      result += np.sum(q_inv_prop*inv_ranking)
    else:
      less_mask = np.less_equal(q_scores[:, None],
                                q_scores[None, :]).astype(np.float64)
      pair_inv_prop = rerank_get(qid, inv_prop)
      result += np.sum(less_mask*pair_inv_prop)
      # result += np.sum((less_mask*2.)*pair_inv_prop.T)
      denom += np.sum(np.diag(pair_inv_prop))

  if rerank:
    result /= denom

  return result

def calc_true_dcg_loss(ranking_model, data_split, cutoff=0):
  all_docs = data_split.feature_matrix
  all_scores = np.dot(all_docs, ranking_model)

  result = 0.
  denom = 0.
  for qid in np.arange(data_split.num_queries()):
    s_i, e_i = data_split.doclist_ranges[qid:qid+2]
    q_scores = all_scores[s_i:e_i]
    q_labels = data_split.query_labels(qid)

    label_filter = np.greater(q_labels, 2)
    inv_ranking = clk.rank_and_invert(q_scores)[1]
    dcg_gain = 1./np.log2(inv_ranking[label_filter]+2.)
    if cutoff > 0:
      dcg_gain[np.greater_equal(inv_ranking[label_filter], cutoff)] = 0.

    result += np.sum(dcg_gain)
    denom += np.sum(label_filter)

  return -result/denom

def estimate_dcg_loss(ranking_model,
                      data_split,
                      clicks,
                      cutoff=0):
  total_clicks = clicks['num_clicks']
  doc_clicks = clicks['clicks_per_doc'].astype(np.float64)
  inv_prop = clicks['average_weights']*(doc_clicks/float(total_clicks))
  inv_prop /= np.sum(inv_prop)
  all_docs = data_split.feature_matrix
  all_scores = np.dot(all_docs, ranking_model)

  rerank = clicks['rerank']
  if rerank:
    rerank_ranges = clicks['rerank_ranges']
    inv_ranking = clicks['inverted_ranking']
    n_doc_per_q = (data_split.doclist_ranges[1:]
                   - data_split.doclist_ranges[:-1])

    def rerank_get(qid, values):
      s_j, e_j = rerank_ranges[qid:qid+2]
      n = n_doc_per_q[qid]
      return np.reshape(values[s_j:e_j], (n, n))

  result = 0.
  denom = 0.
  for qid in clicks['queries']:
    s_i, e_i = data_split.doclist_ranges[qid:qid+2]
    q_scores = all_scores[s_i:e_i]
    if not rerank:
      q_inv_prop = inv_prop[s_i:e_i]
      inv_ranking = clk.rank_and_invert(q_scores)[1]
      dcg_gain = 1./np.log2(inv_ranking+2.)
      if cutoff > 0:
        dcg_gain[np.greater_equal(inv_ranking, cutoff)] = 0.
      result += np.sum(dcg_gain*q_inv_prop)
    else:
      q_prop = rerank_get(qid, inv_prop)
      q_log_rnk = inv_ranking[s_i:e_i]
      less_mask = np.less_equal(q_scores[:, None],
                                q_scores[None, :]).astype(np.float64)
      low_rnk, low_w, up_w = dcg_rerank_weights(
                                                q_log_rnk,
                                                less_mask,
                                                q_prop,
                                                clicks['cutoff'])
      l_dcg_w = 1./np.log2(low_rnk+1.)
      u_dcg_w = 1./np.log2(low_rnk+2.)

      dcg_w = low_w * l_dcg_w + up_w * u_dcg_w

      result += np.sum(dcg_w)
      denom += np.sum(np.diag(q_prop))

  if rerank:
    result /= denom

  return -result

def dcg_estimate_rerank_weights(log_rnk, less_mask, prop, cutoff):
  n_doc = log_rnk.size
  always_included = np.less(log_rnk, cutoff-1)
  sometimes_included = np.logical_not(always_included)
  lower_rank = np.sum(less_mask*always_included[None, :], axis=1)
  lower_rank[sometimes_included] += 1

  added_to_rank = less_mask*sometimes_included[None, :]
  prop_added = np.sum(added_to_rank*prop, axis=1)
  prop_added[sometimes_included] = 0.
  prop_not_added = np.diag(prop) - prop_added

  return lower_rank, prop_not_added, prop_added

def dcg_rerank_weights(log_rnk, less_mask, prop, cutoff):
  n_doc = log_rnk.size
  always_included = np.less(log_rnk, cutoff-1)
  sometimes_included = np.logical_not(always_included)
  lower_rank = np.sum(less_mask*always_included[None, :], axis=1)
  lower_rank[sometimes_included] += 1

  added_to_rank = less_mask[always_included, :]*sometimes_included[None, :]
  prop_added = np.sum(added_to_rank*prop[always_included, :], axis=1) #- np.diag(prop*sometimes_included[None, :])
  prop_not_added = np.diag(prop)[always_included] - prop_added

  up_w = np.zeros((n_doc, n_doc))
  down_w = np.zeros((n_doc, n_doc))
  down_w[always_included, :] = prop_not_added[:, None] * always_included[None, :]
  down_w[sometimes_included, :] = prop[sometimes_included, :]
  up_w[always_included, :] = prop_added[:, None] * always_included[None, :]
  up_w[always_included, :] += prop[always_included, :] * added_to_rank

  return lower_rank, down_w, up_w

def optimize(loss_name,
             data,
             train_clicks,
             validation_clicks,
             learning_rate,
             trial_epochs,
             max_epochs=50,
             epsilon_thres=0.0001,
             learning_rate_decay=0.97,
             cutoff=5):

  if loss_name == 'monotonic':
    est_loss_fn = estimate_dcg_loss
    true_loss_fn = calc_true_dcg_loss
  else:
    est_loss_fn = estimate_loss
    true_loss_fn = calc_true_loss

  total_clicks = train_clicks['num_clicks']
  ave_weights = train_clicks['average_weights']
  doc_clicks = train_clicks['clicks_per_doc'].astype(np.float64)
  train_queries = train_clicks['queries']

  rerank = train_clicks['rerank']
  inv_prop = ave_weights*(doc_clicks/float(total_clicks))
  if not rerank:
    inv_prop /= np.sum(inv_prop)
  else:
    rerank_ranges = train_clicks['rerank_ranges']
    inv_ranking = train_clicks['inverted_ranking']
    n_doc_per_q = (data.train.doclist_ranges[1:]
                   - data.train.doclist_ranges[:-1])

    def rerank_get(qid, values):
      s_j, e_j = rerank_ranges[qid:qid+2]
      n = n_doc_per_q[qid]
      return np.reshape(values[s_j:e_j], (n, n))

    self_norm = 0.
    for qid in train_queries:
      q_prop = rerank_get(qid, inv_prop)
      self_norm += np.sum(np.diag(q_prop))
    inv_prop /= self_norm

  best_model = np.zeros(data.train.datafold.num_features)
  best_loss = np.inf
  best_epoch = 0
  pivot_loss = np.inf
  model = np.zeros(data.train.datafold.num_features)

  start_time = time.time()

  absolute_error = 1.-np.sum(inv_prop)
  print('Normalization error: %s' % absolute_error)

  num_docs = data.train.num_docs()
  doc_feat = data.train.feature_matrix

  epoch_i = 0
  cur_loss = est_loss_fn(model,
                         data.validation,
                         validation_clicks)

  true_loss = true_loss_fn(model,
                           data.validation)
  print(epoch_i, '%0.05f' % cur_loss, '%0.05f' % true_loss)

  stop_epoch = trial_epochs
  while epoch_i < min(stop_epoch, max_epochs):
    permutation = np.random.permutation(train_queries)
    for qid in permutation:
      q_docs = data.train.query_feat(qid)
      q_scores = np.dot(q_docs, model)
      n_docs = q_docs.shape[0]

      less_mask = np.less_equal(q_scores[:, None], q_scores[None, :]+1.).astype(float)
      s_i, e_i = data.train.doclist_ranges[qid:qid+2]
      if not rerank:
        q_prop = inv_prop[s_i:e_i]
        activation_gradient = -less_mask*q_prop[:, None]
      else:
        q_prop = rerank_get(qid, inv_prop)
        activation_gradient = -less_mask*q_prop
      if loss_name == 'monotonic':
        if not rerank:
          up_rank = np.sum(np.maximum(1 - (q_scores[:, None] - q_scores[None, :]), 0), axis=1)
          dcg_weights = 1./(np.log2(up_rank+1.)**2*np.log(2)*(up_rank+1))
          activation_gradient *= dcg_weights[:, None]
        else:
          q_log_rnk = inv_ranking[s_i:e_i]
          low_rnk, low_w, up_w = dcg_rerank_weights(q_log_rnk,
                                                    less_mask,
                                                    q_prop,
                                                    cutoff)
          l_dcg_w = 1./(np.log2(low_rnk+1.)**2*np.log(2)*(low_rnk+1))
          u_dcg_w = 1./(np.log2(low_rnk+2.)**2*np.log(2)*(low_rnk+2))

          dcg_w = low_w * l_dcg_w[:, None] + up_w * u_dcg_w[:, None]
          activation_gradient = -less_mask*dcg_w
      elif loss_name == 'lambdaloss-truncated':
        raise NotImplementedError('Truncated loss needs implementation for optimize.')

      np.fill_diagonal(activation_gradient,
                       np.diag(activation_gradient)
                       - np.sum(activation_gradient, axis=1))

      doc_weights = np.sum(activation_gradient, axis=0)

      gradient = np.sum(q_docs * doc_weights[:, None], axis=0)
      
      model += (learning_rate*gradient
                *(learning_rate_decay**epoch_i))

    epoch_i += 1
    cur_loss = est_loss_fn(model,
                           data.validation,
                           validation_clicks)

    true_loss = true_loss_fn(model,
                             data.validation)
    print(epoch_i, '%0.05f' % cur_loss, '%0.05f' % true_loss)

    if cur_loss < best_loss:
      best_model = model
      best_loss = cur_loss
      best_epoch = epoch_i
      if pivot_loss - cur_loss > epsilon_thres:
        pivot_loss = cur_loss
        stop_epoch = epoch_i + trial_epochs

  true_loss = true_loss_fn(model,
                           data.validation)
  
  result = {
      'model': best_model,
      'estimated_loss': best_loss,
      'true_loss': true_loss,
      'epoch': best_epoch,
      'total_time_spend': time.time()-start_time,
      'time_per_epoch': (time.time()-start_time)/float(epoch_i),
      'learning_rate': learning_rate,
      'trial_epochs': trial_epochs,
      'learning_rate_decay': learning_rate_decay,
    }
  # print(learning_rate, clip_thres, best_epoch, best_loss, true_loss)
  return result

def linear_diff_grad(scores):
  return np.less_equal(scores[:, None], scores[None, :]+1.).astype(float)

def sigmoid_diff_grad(scores):
  score_diff = scores[:, None] - scores[None, :]
  # for stability we cap this at 700
  score_diff = np.maximum(score_diff, -700.)
  exp_diff = np.exp(-score_diff)
  return 1./((1+exp_diff)*np.log(2.))*exp_diff

def optimize_dcg(
             loss_name,
             data,
             train_clicks,
             validation_clicks,
             learning_rate,
             trial_epochs,
             max_epochs=50,
             epsilon_thres=0.0001,
             learning_rate_decay=0.97,
             cutoff=5):

  est_loss_fn = estimate_dcg_loss
  true_loss_fn = calc_true_dcg_loss

  total_clicks = train_clicks['num_clicks']
  ave_weights = train_clicks['average_weights']
  doc_clicks = train_clicks['clicks_per_doc'].astype(np.float64)
  train_queries = train_clicks['queries']

  inv_prop = ave_weights*(doc_clicks/float(total_clicks))
  inv_prop /= np.sum(inv_prop)

  best_model = np.zeros(data.train.datafold.num_features)
  best_loss = np.inf
  best_epoch = 0
  pivot_loss = np.inf
  model = np.zeros(data.train.datafold.num_features)

  start_time = time.time()

  absolute_error = 1.-np.sum(inv_prop)
  print('Normalization error: %s' % absolute_error)

  num_docs = data.train.num_docs()
  doc_feat = data.train.feature_matrix

  epoch_i = 0
  cur_loss = est_loss_fn(model,
                         data.validation,
                         validation_clicks,
                         cutoff)

  true_loss = true_loss_fn(model,
                           data.validation,
                           cutoff)
  print(epoch_i, '%0.05f' % cur_loss, '%0.05f' % true_loss)

  stop_epoch = trial_epochs
  while epoch_i < min(stop_epoch, max_epochs):
    permutation = np.random.permutation(train_queries)
    for qid in permutation:
      q_docs = data.train.query_feat(qid)
      q_scores = np.dot(q_docs, model)
      n_docs = q_docs.shape[0]

      s_i, e_i = data.train.doclist_ranges[qid:qid+2]
      q_prop = inv_prop[s_i:e_i]

      if loss_name in ['monotonic',
                       'log_monotonic',
                       'relevant_rank',
                       'log_relevant_rank']:

        if loss_name in ['monotonic', 'relevant_rank']:
          diff_grad = linear_diff_grad(q_scores)
        elif loss_name in ['log_monotonic', 'log_relevant_rank']:
          diff_grad = sigmoid_diff_grad(q_scores)
        activation_gradient = -diff_grad*q_prop[:, None]
        
        if 'monotonic' in loss_name:
          if loss_name == 'monotonic':
            up_rank = np.sum(np.maximum(1 - (q_scores[:, None] - q_scores[None, :]), 0), axis=1)
          elif loss_name == 'log_monotonic':
            score_diff = q_scores[:, None] - q_scores[None, :]
            up_rank = np.sum(
                np.log2(1 + np.exp(-score_diff)),
              axis=1)

          dcg_weights = 1./(np.log2(up_rank+1.)**2*np.log(2)*(up_rank+1))
          activation_gradient *= dcg_weights[:, None]

      elif loss_name in ['lambdaloss@k', 'lambdaloss-full', 'lambdaloss-truncated']:
        q_inv = clk.rank_and_invert(q_scores)[1]

        prop_diff = q_prop[:, None] - q_prop[None, :]
        prop_mask = np.less_equal(prop_diff, 0.)

        if loss_name == 'lambdaloss-truncated':
          rnk_vec = np.less(q_inv, cutoff)
          rnk_mask = np.logical_or(rnk_vec[:, None],
                                    rnk_vec[None, :])

          prop_mask = np.logical_or(np.logical_not(rnk_mask), prop_mask)

        rank_diff = np.abs(q_inv[:, None] - q_inv[None, :])
        rank_diff[prop_mask] = 1.

        disc_upp = 1. / np.log2(rank_diff+1.)
        disc_low = 1. / np.log2(rank_diff+2.)
        if loss_name == 'lambdaloss@k':
          disc_upp[np.greater(rank_diff, cutoff)] = 0.
          disc_low[np.greater(rank_diff, cutoff-1)] = 0.

        pair_w = disc_upp - disc_low
        pair_w *= np.abs(prop_diff)
        pair_w[prop_mask] = 0.
        
        score_diff = q_scores[:, None] - q_scores[None, :]
        score_diff[prop_mask] = 0.
        safe_diff = np.minimum(-score_diff, 500)
        act = 1./(1 + np.exp(safe_diff))
        act[prop_mask] = 0.
        safe_exp = pair_w - 1.
        safe_exp[prop_mask] = 0.

        log2_grad = 1./(act**pair_w*np.log(2))
        power_grad = pair_w*(act)**safe_exp
        sig_grad = act*(1-act)

        activation_gradient = -log2_grad*power_grad*sig_grad

      np.fill_diagonal(activation_gradient,
                         np.diag(activation_gradient)
                         - np.sum(activation_gradient, axis=1))

      doc_weights = np.sum(activation_gradient, axis=0)

      # print('########')
      # print(q_prop)
      # print(doc_weights)

      gradient = np.sum(q_docs * doc_weights[:, None], axis=0)
      
      model += (learning_rate*gradient
                *(learning_rate_decay**epoch_i))

    epoch_i += 1
    cur_loss = est_loss_fn(model,
                           data.validation,
                           validation_clicks,
                           cutoff)

    true_loss = true_loss_fn(model,
                             data.validation,
                             cutoff)
    print(epoch_i, '%0.05f' % cur_loss, '%0.05f' % true_loss)

    if cur_loss < best_loss:
      best_model = model
      best_loss = cur_loss
      best_epoch = epoch_i
      if pivot_loss - cur_loss > epsilon_thres:
        pivot_loss = cur_loss
        stop_epoch = epoch_i + trial_epochs

  true_loss = true_loss_fn(best_model,
                           data.validation,
                           cutoff)
  
  result = {
      'model': best_model,
      'estimated_loss': best_loss,
      'true_loss': true_loss,
      'epoch': best_epoch,
      'total_time_spend': time.time()-start_time,
      'time_per_epoch': (time.time()-start_time)/float(epoch_i),
      'learning_rate': learning_rate,
      'trial_epochs': trial_epochs,
      'learning_rate_decay': learning_rate_decay,
    }
  # print(learning_rate, clip_thres, best_epoch, best_loss, true_loss)
  return result

def lambdaloss_rerank_weights(log_rnk, less_mask, opt_cutoff, log_cutoff):
  n_doc = log_rnk.size
  always_included = np.less(log_rnk, log_cutoff-1)
  sometimes_included = np.logical_not(always_included)
  lower_rank = np.sum(less_mask*always_included[None, :], axis=1)
  lower_rank[sometimes_included] += 1

  always_ind = np.where(always_included)[0]
  sometimes_ind = np.where(sometimes_included)[0]

  higher_rank = lower_rank[always_ind, None] + less_mask[always_ind[:, None], sometimes_ind[None,  :]]

  lower_distance = np.zeros((n_doc, n_doc))
  lower_distance[always_ind[:, None], sometimes_ind[None,  :]] = higher_rank - lower_rank[None, sometimes_ind]
  # lower_distance[sometimes_ind[:, None], always_ind[None, :]] = higher_rank.T - lower_rank[sometimes_ind, None]
  lower_distance += lower_distance.T

  high_mask = np.less(higher_rank, opt_cutoff)
  low_mask = np.less(lower_rank, opt_cutoff)

  sometimes_mask = np.logical_or(high_mask, low_mask[None, sometimes_ind])

  lower_weight = np.zeros((n_doc, n_doc))
  lower_weight[always_ind[:, None], sometimes_ind[None,  :]] = np.logical_or(high_mask, low_mask[None, sometimes_ind])
  lower_weight += lower_weight.T

  lower_distance[always_ind[:, None], always_ind[None, :]] = lower_rank[always_ind, None] - lower_rank[None, always_ind]
  # lower_distance[always_ind[None, :], always_ind[:, None]] = lower_rank[None, always_ind] - lower_rank[always_ind, None]

  lower_distance = np.abs(lower_distance)

  higher_distance = np.zeros((n_doc, n_doc))
  higher_weight = np.zeros((n_doc, n_doc))

  n_sometimes = np.sum(sometimes_included)
  if n_sometimes  == 0:
    lower_weight = np.logical_or(low_mask[:, None], low_mask[None, :]).astype(np.float64)
  else:
    higher_distance[always_ind[:, None], always_ind[None, :]] = lower_distance[always_ind[:, None], always_ind[None, :]] + 1
    # higher_distance[always_ind[None, :], always_ind[:, None]] = lower_distance[always_ind[None, :], always_ind[:, None]] + 1

    cutoff_check = np.logical_or(high_mask[:, None, :], high_mask[None, :, :])

    less_always = less_mask[always_ind[:, None], sometimes_ind[None,  :]]
    xor_less_always = np.logical_xor(
        less_always[:,None,:], less_always[None, :, :]
      )

    high_count = np.logical_and(cutoff_check, xor_less_always)
    low_count = np.logical_and(cutoff_check, np.logical_not(xor_less_always))

    lower_weight[always_ind[:, None], always_ind[None,  :]] = np.sum(low_count, axis=2)/float(n_sometimes)
    higher_weight[always_ind[:, None], always_ind[None,  :]] = np.sum(high_count, axis=2)/float(n_sometimes)

  return lower_distance, lower_weight, higher_distance, higher_weight

def optimize_rerank_dcg(
             loss_name,
             data,
             train_clicks,
             validation_clicks,
             learning_rate,
             trial_epochs,
             max_epochs=50,
             epsilon_thres=0.0001,
             learning_rate_decay=0.97,
             cutoff=5,
             log_cutoff=5):

  assert loss_name == 'lambdaloss-truncated'
  est_loss_fn = estimate_dcg_loss
  true_loss_fn = calc_true_dcg_loss

  total_clicks = train_clicks['num_clicks']
  ave_weights = train_clicks['average_weights']
  doc_clicks = train_clicks['clicks_per_doc'].astype(np.float64)
  train_queries = train_clicks['queries']

  inv_prop = ave_weights*(doc_clicks/float(total_clicks))
  rerank_ranges = train_clicks['rerank_ranges']
  inv_ranking = train_clicks['inverted_ranking']
  n_doc_per_q = (data.train.doclist_ranges[1:]
                 - data.train.doclist_ranges[:-1])

  def rerank_get(qid, values):
    s_j, e_j = rerank_ranges[qid:qid+2]
    n = n_doc_per_q[qid]
    return np.reshape(values[s_j:e_j], (n, n))

  self_norm = 0.
  for qid in train_queries:
    q_prop = rerank_get(qid, inv_prop)
    self_norm += np.sum(np.diag(q_prop))
  inv_prop /= self_norm

  best_model = np.zeros(data.train.datafold.num_features)
  best_loss = np.inf
  best_epoch = 0
  pivot_loss = np.inf
  model = np.zeros(data.train.datafold.num_features)

  start_time = time.time()

  absolute_error = 1.-np.sum(inv_prop)
  print('Normalization error: %s' % absolute_error)

  num_docs = data.train.num_docs()
  doc_feat = data.train.feature_matrix

  epoch_i = 0
  cur_loss = est_loss_fn(model,
                         data.validation,
                         validation_clicks,
                         cutoff)

  true_loss = true_loss_fn(model,
                           data.validation,
                           cutoff)
  print(epoch_i, '%0.05f' % cur_loss, '%0.05f' % true_loss)

  stop_epoch = trial_epochs
  while epoch_i < min(stop_epoch, max_epochs):
    permutation = np.random.permutation(train_queries)
    for qid in permutation:
      q_prop = rerank_get(qid, inv_prop)
      if np.sum(q_prop) == 0:
        continue
      q_prop = np.diag(q_prop)

      q_docs = data.train.query_feat(qid)
      q_scores = np.dot(q_docs, model)
      n_docs = q_docs.shape[0]

      less_mask = np.less_equal(
                                q_scores[:, None],
                                q_scores[None, :]
                              ).astype(float)
      s_i, e_i = data.train.doclist_ranges[qid:qid+2]

      q_log_rnk = inv_ranking[s_i:e_i]
      low_diff, low_w, up_diff, up_w = lambdaloss_rerank_weights(
                                                  q_log_rnk,
                                                  less_mask,
                                                  cutoff,
                                                  log_cutoff)



      def grad_calc(rank_diff, diff_weights):
        prop_diff = q_prop[:, None] - q_prop[None, :]
        prop_mask = np.less_equal(prop_diff, 0.)
        prop_mask = np.logical_or(prop_mask,
                                  np.less_equal(diff_weights, 0.))

        rank_diff[prop_mask] = 1.

        # tiebreaker
        rank_diff[np.less(rank_diff, 1)] = 1


        disc_upp = 1. / np.log2(rank_diff+1.)
        disc_low = 1. / np.log2(rank_diff+2.)
        if loss_name == 'lambdaloss@k':
          disc_upp[np.greater(rank_diff, cutoff)] = 0.
          disc_low[np.greater(rank_diff, cutoff-1)] = 0.

        pair_w = disc_upp - disc_low
        pair_w *= np.abs(prop_diff)
        pair_w[prop_mask] = 0.
        
        score_diff = q_scores[:, None] - q_scores[None, :]
        score_diff[prop_mask] = 0.
        safe_diff = np.minimum(-score_diff, 500)
        act = 1./(1 + np.exp(safe_diff))
        act[prop_mask] = 0.
        safe_exp = pair_w - 1.
        safe_exp[prop_mask] = 0.

        log2_grad = 1./(act**pair_w*np.log(2))
        power_grad = pair_w*(act)**safe_exp
        sig_grad = act*(1-act)

        return -log2_grad*power_grad*sig_grad

      low_grad = grad_calc(low_diff, low_w)
      up_grad = grad_calc(up_diff, up_w)
      
      activation_gradient = low_w*low_grad + up_w*up_grad

      np.fill_diagonal(activation_gradient,
                         np.diag(activation_gradient)
                         - np.sum(activation_gradient, axis=1))

      doc_weights = np.sum(activation_gradient, axis=0)

      # print('########')
      # print(q_prop)
      # print(doc_weights)

      gradient = np.sum(q_docs * doc_weights[:, None], axis=0)
      
      model += (learning_rate*gradient
                *(learning_rate_decay**epoch_i))

    epoch_i += 1
    cur_loss = est_loss_fn(model,
                           data.validation,
                           validation_clicks,
                           cutoff)

    true_loss = true_loss_fn(model,
                             data.validation,
                             cutoff)
    print(epoch_i, '%0.05f' % cur_loss, '%0.05f' % true_loss)

    if cur_loss < best_loss:
      best_model = model
      best_loss = cur_loss
      best_epoch = epoch_i
      if pivot_loss - cur_loss > epsilon_thres:
        pivot_loss = cur_loss
        stop_epoch = epoch_i + trial_epochs

    if log_cutoff == 1:
      break

  true_loss = true_loss_fn(best_model,
                           data.validation,
                           cutoff)
  
  result = {
      'model': best_model,
      'estimated_loss': best_loss,
      'true_loss': true_loss,
      'epoch': best_epoch,
      'total_time_spend': time.time()-start_time,
      'time_per_epoch': (time.time()-start_time)/float(epoch_i),
      'learning_rate': learning_rate,
      'trial_epochs': trial_epochs,
      'learning_rate_decay': learning_rate_decay,
    }
  # print(learning_rate, clip_thres, best_epoch, best_loss, true_loss)
  return result