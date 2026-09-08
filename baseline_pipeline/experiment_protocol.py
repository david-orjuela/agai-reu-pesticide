"""Pure data/evaluation helpers for matched pesticide-residue experiments."""
from pathlib import Path
import csv
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, r2_score, log_loss)
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     StratifiedShuffleSplit)

CLASS_NAMES = ('low', 'medium', 'high')


def encode_bins(values):
    mapping = {name:i for i,name in enumerate(CLASS_NAMES)}
    normalized = [str(v).strip().lower() for v in values]
    unknown = set(normalized) - set(mapping)
    if unknown:
        raise ValueError(f'Unknown residue bins: {sorted(unknown)}')
    return np.asarray([mapping[v] for v in normalized], dtype=int)


def assign_labels(residue, csv_labels, train_indices, policy='csv',
                  thresholds=None, quantiles=(1/3, 2/3)):
    """Fit boundaries on outer TRAIN labels only; ties go to the lower bin."""
    residue = np.asarray(residue, dtype=float)
    if not np.isfinite(residue).all():
        raise ValueError('Residue targets must all be finite; resolve missing metadata first.')
    if policy == 'csv':
        return np.asarray(csv_labels, dtype=int).copy(), None
    if policy == 'fixed':
        edges = np.asarray(thresholds if thresholds is not None else [], dtype=float)
    elif policy == 'train_quantile':
        q = np.asarray(quantiles, dtype=float)
        if q.shape != (2,) or not (0 < q[0] < q[1] < 1):
            raise ValueError('Specify two increasing quantiles strictly between zero and one.')
        edges = np.quantile(residue[np.asarray(train_indices)], q)
    else:
        raise ValueError(f'Unknown bin policy: {policy}')
    if edges.shape != (2,) or not np.isfinite(edges).all() or edges[0] >= edges[1]:
        raise ValueError('Specify two finite, strictly increasing residue thresholds.')
    return np.searchsorted(edges, residue, side='left'), edges


def make_outer_splits(csv_labels, folds, seed, groups=None):
    """Always stratify on ORIGINAL CSV bins so threshold experiments share folds."""
    y = np.asarray(csv_labels)
    x = np.zeros(len(y))
    if groups is None:
        if np.bincount(y, minlength=3).min() < folds:
            raise ValueError('Not enough original samples in every bin for requested folds.')
        splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    else:
        groups = np.asarray(groups)
        if len(np.unique(groups)) < folds:
            raise ValueError('Not enough independent groups for requested folds.')
        splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(x, y, groups))
    check_partition(splits, len(y), groups)
    return splits


def check_partition(splits, n, groups=None):
    held_out = []
    for train,test in splits:
        train,test = np.asarray(train,dtype=int),np.asarray(test,dtype=int)
        if len(np.unique(train)) != len(train) or len(np.unique(test)) != len(test):
            raise ValueError('Duplicate indices in a split.')
        if np.intersect1d(train,test).size or set(np.r_[train,test]) != set(range(n)):
            raise ValueError('Each fold must partition the full dataset with no overlap.')
        if groups is not None and set(np.asarray(groups)[train]) & set(np.asarray(groups)[test]):
            raise ValueError('Groups overlap between train and outer test.')
        held_out.extend(test)
    if sorted(held_out) != list(range(n)):
        raise ValueError('Each sample must appear in exactly one outer test fold per seed.')


def make_inner_split(outer_train, labels, seed, groups=None, fraction=.2, group_folds=4):
    outer_train = np.asarray(outer_train)
    y = np.asarray(labels)[outer_train]
    if groups is None:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        a,b = next(splitter.split(np.zeros(len(y)),y))
    else:
        g = np.asarray(groups)[outer_train]
        if len(np.unique(g)) < group_folds:
            raise ValueError('Too few outer-training groups for the configured inner folds.')
        splitter = StratifiedGroupKFold(group_folds,shuffle=True,random_state=seed)
        a,b = next(splitter.split(np.zeros(len(y)),y,g))
        assert not set(g[a]) & set(g[b])
    if len(np.unique(y[a])) != 3:
        raise ValueError('An inner training bin is empty; revise split/threshold design.')
    return outer_train[a],outer_train[b]


def classification_metrics(y, prediction, probabilities):
    y,prediction = np.asarray(y,dtype=int),np.asarray(prediction,dtype=int)
    p = np.asarray(probabilities,dtype=float)
    if p.shape != (len(y),3) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Expected finite, nonnegative classification probabilities (N,3).')
    if not np.allclose(p.sum(1),1,atol=1e-5):
        raise ValueError('Classification probabilities do not sum to one.')
    p = p / p.sum(1,keepdims=True)
    result = {'accuracy':float(np.mean(y==prediction)),
              'macro_f1':float(f1_score(y,prediction,labels=[0,1,2],average='macro',zero_division=0)),
              'log_loss':float(log_loss(y,p,labels=[0,1,2])),
              'brier':float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))),
              'ordinal_mae':float(np.abs(y-prediction).mean()),
              'extreme_error_rate':float((np.abs(y-prediction)==2).mean())}
    for metric,func in [('recall',recall_score),('precision',precision_score),('f1',f1_score)]:
        for name,value in zip(CLASS_NAMES,func(y,prediction,labels=[0,1,2],average=None,zero_division=0)):
            result[f'{metric}_{name}'] = float(value)
    for i,name in enumerate(CLASS_NAMES):
        result[f'support_{name}']=int((y==i).sum())
        result[f'predicted_{name}']=int((prediction==i).sum())
    return result


def regression_metrics(y,prediction):
    y,p = np.asarray(y,dtype=float),np.asarray(prediction,dtype=float)
    residual = y-p
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Nonfinite regression target or prediction.')
    nonconstant = len(y)>1 and np.std(y)>0 and np.std(p)>0
    return {'mae':float(np.abs(residual).mean()),'rmse':float(np.sqrt(np.mean(residual**2))),
            'r2':float(r2_score(y,p)) if len(y)>1 and np.std(y)>0 else float('nan'),
            'pearson_r':float(pearsonr(y,p).statistic) if nonconstant else float('nan'),
            'spearman_r':float(spearmanr(y,p).statistic) if nonconstant else float('nan'),
            'residual_mean':float(residual.mean()),'target_std':float(y.std()),
            'prediction_std':float(p.std()),'negative_predictions':int((p<0).sum())}


def aggregate_views(values, task):
    """(views,N,classes) probabilities or (views,N) regression predictions."""
    values=np.asarray(values)
    if task == 'icp_regression':
        return {'mean':values.mean(0)},values.std(0)
    mean=values.mean(0)
    votes=np.eye(3)[values.argmax(-1)].sum(0)
    # Break vote ties by mean probability among tied classes, then lower ID.
    tied=votes==votes.max(-1,keepdims=True)
    voted=np.where(tied,mean,-np.inf).argmax(-1)
    disagreement=1-votes.max(-1)/len(values)
    return {'probabilities':mean,'mean':mean.argmax(-1),'vote':voted},disagreement


def write_csv(path, rows):
    if not rows:
        return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def distribution_rows(indices,labels,residue,seed,fold,role):
    indices=np.asarray(indices)
    rows=[]
    for k,name in enumerate(CLASS_NAMES):
        values=np.asarray(residue)[indices[np.asarray(labels)[indices]==k]]
        rows.append({'seed':seed,'fold':fold,'role':role,'class':name,'n':len(values),
                     'residue_min':float(values.min()) if len(values) else None,
                     'residue_max':float(values.max()) if len(values) else None,
                     'residue_mean':float(values.mean()) if len(values) else None,
                     'residue_std':float(values.std()) if len(values) else None})
    return rows
