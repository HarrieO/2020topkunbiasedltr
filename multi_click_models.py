# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import dataset
import numpy as np

def included_queries(data_split, squashed_clicks, rerank):
  if rerank:
    doc_per_q = (data_split.doclist_ranges[1:]
                 - data_split.doclist_ranges[:-1])
    rerank_weight_ranges = np.concatenate([[0], doc_per_q])
    rerank_weight_ranges = np.cumsum(rerank_weight_ranges**2)
    num_queries = data_split.num_queries()
    q_mask = np.zeros(num_queries, dtype=np.bool)
    for qid in range(num_queries):
      s_i, e_i = rerank_weight_ranges[qid:qid+2]
      q_mask[qid] = np.sum(squashed_clicks[s_i:e_i]) > 0
  else:
    summed_weights = np.cumsum(squashed_clicks)
    q_sum = summed_weights[data_split.doclist_ranges[1:]-1]
    q_mask = np.greater(np.diff(q_sum), 0)
    first_n = summed_weights[data_split.doclist_ranges[1]-1]
    q_mask = np.concatenate(([first_n > 0], q_mask))
  return q_mask

def read_model(model_file_path, data, scale=1.0):
  model = np.zeros(data.num_features)
  with open(model_file_path, 'r') as model_file:
    model_line = model_file.readlines()[-1]
    model_line = model_line[:model_line.find('#')]
    for feat_tuple in model_line.split()[1:]:
      f_i, f_v = feat_tuple.split(':')
      f_i = data.inverse_feature_map[int(f_i)]
      model[f_i] = float(f_v)
  if np.linalg.norm(model) > 0:
    model /= np.linalg.norm(model)/scale
  return model

def sample_from_click_probs(click_probs):
  coin_flips = np.random.uniform(size=click_probs.shape)
  return np.where(np.less(coin_flips, click_probs))[0]

def rank_and_invert(scores):
  n_docs = scores.shape[0]
  rank_ind = np.argsort(scores)[::-1]
  inverted = np.empty(n_docs, dtype=rank_ind.dtype)
  inverted[rank_ind] = np.arange(n_docs)
  return rank_ind, inverted

def generate_clicks(data_split,
                    ranking_model,
                    click_model,
                    n_clicks,
                    cutoff,
                    eta):

  def inverse_rank_prop(inv_ranking, cutoff):
    result = (1./(inv_ranking+1.))**eta
    if cutoff > 0:
      result[inv_ranking>=cutoff] = 0.
    return result

  if click_model == 'binarized':
    def relevance_click_prob(labels):
      n_docs = labels.shape[0]
      rel_prob = np.full(n_docs, 0.1)
      rel_prob[labels>2] = 1.
      return rel_prob
  elif click_model == 'noisybinarized':
    def relevance_click_prob(labels):
      n_docs = labels.shape[0]
      rel_prob = np.full(n_docs, 0.1)
      rel_prob[labels>2] = .15
      return rel_prob
  elif click_model == 'linear':
    def relevance_click_prob(labels):
      n_docs = labels.shape[0]
      max_click_prob = 1.
      min_click_prob = 0.1
      click_prob_step = (max_click_prob-min_click_prob)/4.
      rel_prob = np.full(n_docs, min_click_prob)
      rel_prob += click_prob_step*labels
      return rel_prob
  else:
    raise ValueError('Unknown click model: %s' % click_model)

  max_len = np.amax(data_split.doclist_ranges[1:]
                    - data_split.doclist_ranges[:-1])

  no_cutoff_i = 0
  no_cutoff_result = {
      'qid': np.empty(n_clicks, dtype=np.int64),
      'clicked': np.empty(n_clicks, dtype=np.int64),
      'prop': np.empty(n_clicks),
  }
  cutoff_det_i = 0
  cutoff_det_result = {
      'qid':      np.empty(n_clicks, dtype=np.int64),
      'clicked':  np.empty(n_clicks, dtype=np.int64),
      'prop':     np.empty(n_clicks),
      'included': np.empty((n_clicks, cutoff), dtype=np.int64),
    }
  replace_last_i = 0
  cutoff_obs_result = {
      'qid':      np.empty(n_clicks, dtype=np.int64),
      'clicked':  np.empty(n_clicks, dtype=np.int64),
      'prop':     np.empty(n_clicks),
      'included': np.empty((n_clicks, cutoff), dtype=np.int64),
    }
  cutoff_rep_result = {
      'qid':      np.empty(n_clicks, dtype=np.int64),
      'clicked':  np.empty(n_clicks, dtype=np.int64),
      'prop':     np.empty(n_clicks),
    }

  all_docs = data_split.feature_matrix
  all_scores = np.dot(all_docs, ranking_model)

  num_queries_sampled = 0
  while min(no_cutoff_i,
            cutoff_det_i,
            replace_last_i) < n_clicks:
    num_queries_sampled += 1
    qid = np.random.choice(data_split.num_queries())
    s_i, e_i = data_split.doclist_ranges[qid:qid+2]
    n_docs = e_i - s_i
    q_scores = all_scores[s_i:e_i]
    q_labels = data_split.query_labels(qid)

    all_rel = relevance_click_prob(q_labels)

    full_rank, full_inv = rank_and_invert(q_scores)

    if no_cutoff_i < n_clicks:
      prop = inverse_rank_prop(full_inv, 0)
      for c_i in sample_from_click_probs(all_rel*prop):
        no_cutoff_result['qid'][no_cutoff_i] = qid
        no_cutoff_result['clicked'][no_cutoff_i] = c_i
        no_cutoff_result['prop'][no_cutoff_i] = prop[c_i]
        no_cutoff_i += 1
        if no_cutoff_i >= n_clicks:
          break

    if cutoff_det_i < n_clicks:
      prop = inverse_rank_prop(full_inv, cutoff)
      cur_included = np.where(np.greater(prop, 0))[0]
      cutoff_diff = cutoff - cur_included.shape[0]
      if cutoff_diff > 0:
        cur_included = np.concatenate(
          (cur_included,
           np.repeat(cur_included[0],
                     cutoff_diff)),
          axis=0)
      for c_i in sample_from_click_probs(all_rel*prop):
        cutoff_det_result['qid'][cutoff_det_i] = qid
        cutoff_det_result['clicked'][cutoff_det_i] = c_i
        cutoff_det_result['prop'][cutoff_det_i] = prop[c_i]
        cutoff_det_result['included'][cutoff_det_i, :] = cur_included
        cutoff_det_i += 1
        if cutoff_det_i >= n_clicks:
          break

    if replace_last_i < n_clicks:
      cut_rank = full_rank.copy()
      cut_inv = full_inv.copy()
      if cutoff < n_docs:
        inc_doc = np.random.choice(cut_rank[cutoff-1:])
        swp_doc = cut_rank[cutoff-1]
        cut_inv[swp_doc] = cut_inv[inc_doc]
        cut_inv[inc_doc] = cutoff-1

      prop = inverse_rank_prop(cut_inv, cutoff)  
      cur_included = np.where(np.greater(prop, 0))[0]
      cutoff_diff = cutoff - cur_included.shape[0]
      if cutoff_diff > 0:
        cur_included = np.concatenate(
          (cur_included,
           np.repeat(cur_included[0],
                     cutoff_diff)),
          axis=0)
      for c_i in sample_from_click_probs(all_rel*prop):
        cutoff_obs_result['qid'][replace_last_i] = qid
        cutoff_obs_result['clicked'][replace_last_i] = c_i
        cutoff_obs_result['prop'][replace_last_i] = prop[c_i]
        cutoff_obs_result['included'][replace_last_i, :] = cur_included

        n_outside = max(n_docs-cutoff+1,1)
        cutoff_rep_result['qid'][replace_last_i] = qid
        cutoff_rep_result['clicked'][replace_last_i] = c_i
        if cutoff < n_docs and c_i == inc_doc:
          cutoff_rep_result['prop'][replace_last_i] = prop[c_i]/float(n_outside)
        else:
          cutoff_rep_result['prop'][replace_last_i] = prop[c_i]

        replace_last_i += 1
        if replace_last_i >= n_clicks:
          break

  return {
    'deterministic': {
      'num_queries_sampled': num_queries_sampled,
      'data_split_name': data_split.name,
      'qid': no_cutoff_result['qid'],
      'clicked': no_cutoff_result['clicked'],
      'prop': no_cutoff_result['prop'],
      'cutoff': 0,
    },
    'deterministic_cutoff': {
      'num_queries_sampled': num_queries_sampled,
      'data_split_name': data_split.name,
      'qid': cutoff_det_result['qid'],
      'clicked': cutoff_det_result['clicked'],
      'prop': cutoff_det_result['prop'],
      'included': cutoff_det_result['included'],
      'cutoff': cutoff,
    },
    'replacelast_oblivious': {
      'num_queries_sampled': num_queries_sampled,
      'data_split_name': data_split.name,
      'qid': cutoff_obs_result['qid'],
      'clicked': cutoff_obs_result['clicked'],
      'prop': cutoff_obs_result['prop'],
      'included': cutoff_obs_result['included'],
      'cutoff': cutoff,
    },
    'replacelast_policyaware': {
      'num_queries_sampled': num_queries_sampled,
      'data_split_name': data_split.name,
      'qid': cutoff_rep_result['qid'],
      'clicked': cutoff_rep_result['clicked'],
      'prop': cutoff_rep_result['prop'],
      'cutoff': cutoff,
    },
  }

def generate_squashed_clicks(logging_policy,
                             data_split,
                             ranking_model,
                             click_model,
                             n_clicks,
                             cutoff,
                             eta,
                             clipping_thres):

  def inverse_rank_prop(inv_ranking, cutoff):
    result = (1./(inv_ranking+1.))**eta
    if cutoff > 0:
      result[inv_ranking>=cutoff] = 0.
    return result

  if click_model == 'binarized':
    def relevance_click_prob(labels):
      n_docs = labels.shape[0]
      rel_prob = np.full(n_docs, 0.1)
      rel_prob[labels>2] = 1.
      return rel_prob
  else:
    raise ValueError('Unknown click model: %s' % click_model)

  rerank = 'rerank' in logging_policy
  if rerank:
    max_len = np.amax(data_split.doclist_ranges[1:]
                      - data_split.doclist_ranges[:-1])
    ave_weights = np.zeros((data_split.num_docs(), max_len))
    clicks_per_doc = np.zeros((data_split.num_docs(), max_len),
                              dtype=np.int64)
    doc_per_q = (data_split.doclist_ranges[1:]
                 - data_split.doclist_ranges[:-1])
    n_weights = np.sum(doc_per_q**2)
    ave_weights = np.zeros(n_weights)
    clicks_per_doc = np.zeros(n_weights, dtype=np.int64)
    rerank_weight_ranges = np.concatenate([[0], doc_per_q])
    rerank_weight_ranges = np.cumsum(rerank_weight_ranges**2)
  else:
    ave_weights = np.zeros(data_split.num_docs())
    clicks_per_doc = np.zeros(data_split.num_docs(),
                              dtype=np.int64)

  all_docs = data_split.feature_matrix
  all_scores = np.dot(all_docs, ranking_model)

  normal_ranking = np.zeros(data_split.num_docs(),
                            dtype=np.int64)
  inverted_ranking = np.zeros(data_split.num_docs(),
                            dtype=np.int64)

  rel_click_prob = np.zeros(data_split.num_docs(),
                            dtype=np.float64)
  for qid in np.arange(data_split.num_queries()):
    s_i, e_i = data_split.doclist_ranges[qid:qid+2]
    n_docs = e_i - s_i
    q_scores = all_scores[s_i:e_i]
    (normal_ranking[s_i:e_i],
     inverted_ranking[s_i:e_i]) = rank_and_invert(q_scores)
    q_labels = data_split.query_labels(qid)
    rel_click_prob[s_i:e_i] = relevance_click_prob(q_labels)

  clip_after = clipping_thres > 0
  clicks_generated = 0
  num_queries_sampled = 0
  while clicks_generated < n_clicks:
    num_queries_sampled += 1
    qid = np.random.choice(data_split.num_queries())
    s_i, e_i = data_split.doclist_ranges[qid:qid+2]
    n_docs = e_i - s_i
    
    rel_prob = rel_click_prob[s_i:e_i]
    norm_rank = normal_ranking[s_i:e_i]
    inv_rank = inverted_ranking[s_i:e_i]

    if 'deterministic' in logging_policy:
      prop = inverse_rank_prop(inv_rank, cutoff)
      c_i = sample_from_click_probs(rel_prob*prop)
      d_i =  c_i + s_i

      if not rerank:
        clicks_per_doc[d_i] += 1
        ave_weights[d_i] = 1./prop[c_i]
      elif c_i.size > 0:
        inc = norm_rank[:cutoff]
        s_j, e_j = rerank_weight_ranges[qid:qid+2]
        cur_weights = np.reshape(ave_weights[s_j:e_j], (n_docs, n_docs))
        cur_clicks = np.reshape(clicks_per_doc[s_j:e_j], (n_docs, n_docs))
        cur_clicks[c_i[:, None], inc[None, :]] += 1
        cur_weights[c_i[:, None], inc[None, :]] = 1./prop[c_i, None]

      clicks_generated += c_i.size

    elif 'replacelast' in logging_policy:
      if cutoff < n_docs:
        inc_doc = np.random.choice(norm_rank[cutoff-1:])
        swp_doc = norm_rank[cutoff-1]
        inv_rank[swp_doc] = inv_rank[inc_doc]
        inv_rank[inc_doc] = cutoff-1
        norm_rank[cutoff-1] = inc_doc
        norm_rank[inv_rank[swp_doc]] = swp_doc

      prop = inverse_rank_prop(inv_rank, cutoff)
      c_i = sample_from_click_probs(rel_prob*prop)
      d_i =  c_i + s_i
      if not rerank:
        clicks_per_doc[d_i] += 1
      else:
        inc = norm_rank[:cutoff]
        s_j, e_j = rerank_weight_ranges[qid:qid+2]
        cur_weights = np.reshape(ave_weights[s_j:e_j], (n_docs, n_docs))
        cur_clicks = np.reshape(clicks_per_doc[s_j:e_j], (n_docs, n_docs))
        cur_clicks[c_i[:, None], inc[None, :]] += 1
      clicks_generated += c_i.size

      n_outside = max(n_docs-cutoff+1,1)
      if cutoff < n_docs and 'oblivious' not in logging_policy:
        denom = np.ones(c_i.shape)
        denom[np.greater_equal(inv_rank[c_i], cutoff-1)] = n_outside
        ave_weights[d_i] = denom/prop[c_i]
      else:
        if not rerank:
          ave_weights[d_i] = 1./prop[c_i]
        else:
          cur_weights[c_i[:, None], inc[None, :]] = 1./prop[c_i, None]

  if clip_after:
    ave_weights = np.minimum(ave_weights, clipping_thres)

  query_mask = included_queries(data_split, clicks_per_doc, rerank)
  queries = np.arange(data_split.num_queries())[query_mask]
  result = {
          'rerank': rerank,
          'num_queries_sampled': num_queries_sampled,
          'data_split_name': data_split.name,
          'average_weights': ave_weights,
          'clicks_per_doc': clicks_per_doc,
          'num_clicks': clicks_generated,
          'cutoff': cutoff,
          'queries': queries,
        }
  if rerank:
    result.update({
      'rerank_ranges': rerank_weight_ranges,
      'inverted_ranking': inverted_ranking,
      })
  return result
