from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pickle

def compute_auroc(id_pps, ood_pps, normalize=False, return_curve=False):
    y = np.concatenate((np.ones_like(ood_pps), np.zeros_like(id_pps)))
    scores = np.concatenate((ood_pps, id_pps))
    if normalize:
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    if return_curve:
        return roc_curve(y, scores)
    else:
        return 100*roc_auc_score(y, scores)

def compute_far(id_pps, ood_pps, rate=5):
    incorrect = len(id_pps[id_pps > np.percentile(ood_pps, rate)])
    return 100*incorrect / len(id_pps)

def compute_px(ppl, lls):
    lengths = np.array([len(ll) for ll in lls])
    logpx = np.log(ppl) * lengths * -1
    return logpx

def compute_ppl(logpx, lls):
    lengths = np.array([len(ll) for ll in lls])
    log_ppl = - logpx / lengths
    return np.exp(log_ppl)

def compute_conditional_prior(pps_labels, lls_labels, probs):
    # p(x) = \sum_y p(x|y) p(y)
    # log p(x) = \sum_y log p(x|y) + log p(y)
    px_labels = {}
    combined_px = None
    for label in pps_labels:
        px_labels[label] = compute_px(pps_labels[label], lls_labels[label])
        if combined_px is None:
            combined_px = px_labels[label] + np.log(probs[label])
        else:
            combined_px += px_labels[label] + np.log(probs[label])

    combined_pps = compute_ppl(combined_px, lls_labels[0])
    return combined_pps, combined_px

def compute_conditional(pps_labels, lls_labels, probs):
    # p(x) = \sum_y p(x|y) p(y|x)
    # log p(x) = \sum_y log p(x|y) + log p(y|x)
    px_labels = {}
    combined_px = None
    for label in pps_labels:
        px_labels[label] = compute_px(pps_labels[label], lls_labels[label])
        if combined_px is None:
            combined_px = px_labels[label] + np.log(probs[:, label])
        else:
            combined_px += px_labels[label] + np.log(probs[:, label])

    combined_pps = compute_ppl(combined_px, lls_labels[0])
    return combined_pps, combined_px

def compute_lm_metric(id_pps, id_lls, ood_pps, ood_lls, id_px=None, ood_px=None, metric='auroc', do_print=False, conditional=False):
    if metric == 'auroc':
        compute_fn = compute_auroc
    else:
        compute_fn = compute_far

    if id_px is None:
        id_px = compute_px(id_pps, id_lls)
    if ood_px is None:
        ood_px = compute_px(ood_pps, ood_lls)

    score_px = compute_fn(-id_px, -ood_px)
    score_ppl = compute_fn(id_pps, ood_pps)
    if do_print:
        if conditional:
            ctext = 'Conditional '
        else:
            ctext = ''
        print(f"{ctext}P(x): {score_px:.3f}")
        print(f"{ctext}Perplexity: {score_ppl:.3f}")
    scores = {
        'p_x': score_px,
        'ppl': score_ppl
    }
    return scores

def compute_auroc_all(id_msp, id_px, id_ppl, ood_msp, ood_px, ood_ppl, do_print=False):
    score_px = compute_auroc(-id_px, -ood_px)
    score_py = compute_auroc(-id_msp, -ood_msp)
    score_ppl = compute_auroc(id_ppl, ood_ppl)
    if do_print:
        print(f"P(x): {score_px:.3f}")
        print(f"P(y | x): {score_py:.3f}")
        print(f"Perplexity: {score_ppl:.3f}")
    scores = {
        'p_x': score_px,
        'p_y': score_py,
        'ppl': score_ppl
    }
    return scores

def compute_metric_all_old(id_pps, id_lls, id_msp, ood_pps, ood_lls, ood_msp, metric='auroc', do_print=False):
    id_px = compute_px(id_pps, id_lls)
    ood_px = compute_px(ood_pps, ood_lls)
    if metric == 'auroc':
        score_px = compute_auroc(-id_px, -ood_px)
        score_py = compute_auroc(-id_msp, -ood_msp)
        score_ppl = compute_auroc(id_pps, ood_pps)
    elif metric == 'far':
        score_px = compute_far(-id_px, -ood_px)
        score_py = compute_far(-id_msp, -ood_msp)
        score_ppl = compute_far(id_pps, ood_pps)
    else:
        raise Exception('Invalid metric name')

    if do_print:
        print(f"Metric {metric}:")
        print(f"P(x): {score_px:.3f}")
        print(f"P(y | x): {score_py:.3f}")
        print(f"Perplexity: {score_ppl:.3f}\n")

    scores = {
        'p_x': score_px,
        'p_y': score_py,
        'ppl': score_ppl
    }
    return scores

def compute_metric_all(id_pps, id_lls, id_msp, id_pps_cond, id_lls_cond,
                        ood_pps, ood_lls, ood_msp, ood_pps_cond, ood_lls_cond,
                        metric='auroc', do_print=False):
    if metric == 'auroc':
        compute_fn = compute_auroc
    else:
        compute_fn = compute_far
    
    scores_lm = compute_lm_metric(id_pps, id_lls, ood_pps, ood_lls, metric=metric, do_print=do_print)
    if id_pps_cond is not None:
        scores_lm_cond = compute_lm_metric(id_pps_cond, id_lls_cond, ood_pps_cond, ood_lls_cond, metric=metric, do_print=do_print, conditional=True)
    else:
        scores_lm_cond = None

    score_py = compute_fn(-id_msp, -ood_msp)

    if do_print:
        print(f"P(y | x): {score_py:.3f}")

    scores = {
        'p_x': scores_lm['p_x'],
        'ppl': scores_lm['ppl'],
        'p_y': score_py
    }

    if scores_lm_cond is not None:
        scores['p_x_cond'] = scores_lm_cond['p_x']
        scores['ppl_cond'] = scores_lm_cond['ppl']

    return scores

def read_model_out(fname):
    if '.pkl' in fname:
        with open(fname, 'rb') as f:
            return pickle.load(f)
    elif '.npy' in fname:
        return np.load(fname)
    else:
        raise KeyError(f'{ftype} not supported for {fname}')
