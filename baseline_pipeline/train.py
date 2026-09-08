"""
AGAI_REU_2026
David Orjuela
Undergraduate Student under Dr. Chen
Training orchestration for pesticide-residue experiments.

Change CONFIG.task, CONFIG.backbone, and CONFIG.random_seeds to select an
experiment. Training, evaluation, plotting, and checkpoint logic adapts
automatically.

Matched residue experiments with inner validation and checkpoint-only TTA.
Run: python -m baseline_pipeline.train --help
"""

import argparse
import hashlib
import json
import random
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from baseline_pipeline.baseline_model import frozen_resnet, frozen_dinov3
from baseline_pipeline.dataset import agai_correct_v3
from baseline_pipeline.experiment_protocol import (
    CLASS_NAMES, encode_bins, assign_labels, make_outer_splits, make_inner_split,
    check_partition, classification_metrics, regression_metrics, aggregate_views,
    write_csv, distribution_rows,
)


class Task(str, Enum):
    BIN_CLASSIFICATION='bin_classification'
    ICP_REGRESSION='icp_regression'


class Backbone(str, Enum):
    RESNET50='resnet50'
    DINOV3='dinov3'


@dataclass(frozen=True)
class ExperimentConfig:
    task: Task = Task.ICP_REGRESSION
    backbone: Backbone = Backbone.DINOV3
    csv_path: str = '/home/davidorjuela/dev/agai-reu-pesticide/datasets/agai_correct/batch_2_v1/master_icp.csv'
    residue_column: str = 'mg_cm2'
    residue_bin_column: str = 'residue_bin'
    sample_id_column: str = 'sample_id'
    group_column: Optional[str] = None
    class_names: Tuple[str,...] = CLASS_NAMES
    dinov3_repo_dir: str = '/home/davidorjuela/dev/dinov3'
    dinov3_weights_path: str = '/home/davidorjuela/dev/agai-reu-pesticide/checkpoints/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'
    preprocessing: str = 'auto'
    augmentation: str = 'hflip'  # none | hflip | d4 | medium_d4
    bin_policy: str = 'csv'  # csv | fixed | train_quantile
    thresholds: Optional[Tuple[float,float]] = None
    quantiles: Tuple[float,float] = (1/3,2/3)
    selection: str = 'inner_refit'  # inner_refit | fixed | legacy_outer
    inner_fraction: float = .2
    inner_group_folds: int = 4
    freeze_resnet_bn: bool = True
    learning_rate: float = .001
    momentum: float = .9
    batch_size: int = 4
    epochs: int = 30
    folds: int = 4
    random_seeds: Tuple[int,...] = (42,)
    num_workers: int = 0
    inference_modes: Tuple[str,...] = ('single','d4')
    tta_scales: Tuple[float,...] = (1.,1.25)
    evaluate_only: bool = False
    source_checkpoint_dir: Path = Path('checkpoints')
    output_root: Path = Path('results/experiments')
    run_name: Optional[str] = None

    @property
    def num_outputs(self):
        return 3 if self.task == Task.BIN_CLASSIFICATION else 1


CONFIG = ExperimentConfig()
MEAN=(.485,.456,.406)
STD=(.229,.224,.225)


class EnsureRGB:
    def __call__(self,image):
        return image.convert('RGB')


def preprocessing_name(config):
    if config.preprocessing == 'auto':
        return 'dinov3_256' if config.backbone == Backbone.DINOV3 else 'legacy_224'
    return config.preprocessing


def make_transform(config,scale=1.):
    mode=preprocessing_name(config)
    size=int(round((256 if mode=='dinov3_256' else 224)*scale/16))*16
    if size<16:raise ValueError('Invalid image size.')
    spatial=([transforms.Resize((size,size),antialias=True)] if mode=='dinov3_256'
             else [transforms.Resize(round(256*size/224)),transforms.CenterCrop(size)])
    return transforms.Compose([EnsureRGB(),*spatial,transforms.ToTensor(),transforms.Normalize(MEAN,STD)])


class ExperimentDataset(Dataset):
    def __init__(self,base,indices,labels,residue,task,augmentation='none'):
        self.base,self.indices=base,np.asarray(indices,dtype=int)
        self.labels,self.residue,self.task=labels,residue,task
        self.augmentation=augmentation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,position):
        i=self.indices[position]
        image,_=self.base[int(i)]
        if self.augmentation in ('hflip','medium_d4') and torch.rand(()).item()<.5:
            image=torch.flip(image,[-1])
        if self.augmentation=='d4' or (self.augmentation=='medium_d4' and self.labels[i]==1):
            image=torch.rot90(image,int(torch.randint(4,()).item()),[-2,-1])
            if self.augmentation=='d4' and torch.rand(()).item()<.5:image=torch.flip(image,[-1])
        target=(torch.tensor(self.labels[i],dtype=torch.long) if self.task==Task.BIN_CLASSIFICATION
                else torch.tensor(self.residue[i],dtype=torch.float32))
        return image.contiguous(),target


def build_resnet(config):
    return frozen_resnet(num_bins=config.num_outputs)


def build_dinov3(config):
    return frozen_dinov3(num_bins=config.num_outputs,repo_dir=config.dinov3_repo_dir,
                         weights_path=config.dinov3_weights_path)


MODEL_BUILDERS={Backbone.RESNET50:build_resnet,Backbone.DINOV3:build_dinov3}


def build_model(config,device):
    model=MODEL_BUILDERS[config.backbone](config).to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,} total; '
          f'{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable')
    return model


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def loader(base,idx,labels,residue,config,seed,training=False):
    data=ExperimentDataset(base,idx,labels,residue,config.task,
                           config.augmentation if training else 'none')
    return DataLoader(data,batch_size=config.batch_size,shuffle=training,
                      num_workers=config.num_workers,generator=torch.Generator().manual_seed(seed))


def run_epoch(model,data,config,device,optimizer=None):
    training=optimizer is not None
    model.train(training)
    if config.backbone==Backbone.RESNET50 and config.freeze_resnet_bn:
        # Frozen parameters alone do not freeze BatchNorm running statistics.
        model.eval()
        model.fc.train(training)
    criterion=nn.CrossEntropyLoss() if config.task==Task.BIN_CLASSIFICATION else nn.SmoothL1Loss()
    y,p,prob=[],[],[]
    loss_sum=0.
    with torch.set_grad_enabled(training):
        for images,target in data:
            images,target=images.to(device),target.to(device)
            if training:optimizer.zero_grad(set_to_none=True)
            output=model(images)
            if config.task==Task.ICP_REGRESSION:output=output.reshape(-1)
            loss=criterion(output,target)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss; inspect targets and optimizer.')
            if training:
                loss.backward()
                optimizer.step()
            y.extend(target.detach().cpu().tolist())
            if config.task==Task.BIN_CLASSIFICATION:
                probabilities=output.softmax(1)
                prob.extend(probabilities.detach().cpu().tolist())
                p.extend(probabilities.argmax(1).detach().cpu().tolist())
            else:p.extend(output.detach().cpu().tolist())
            loss_sum+=loss.item()*len(images)
    metrics=(classification_metrics(y,p,prob) if config.task==Task.BIN_CLASSIFICATION
             else regression_metrics(y,p))
    metrics['loss']=loss_sum/len(data.dataset)
    return metrics


def fit_model(train_idx,validation_idx,base,labels,residue,config,device,seed,stage,history,fixed_epochs=None):
    set_random_seed(seed)
    model=build_model(config,device)
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                              lr=config.learning_rate,momentum=config.momentum)
    train_loader=loader(base,train_idx,labels,residue,config,seed,True)
    val_loader=None if validation_idx is None else loader(base,validation_idx,labels,residue,config,seed)
    best,best_state,best_epoch=float('inf'),None,0
    for epoch in range(1,(fixed_epochs or config.epochs)+1):
        train_metrics=run_epoch(model,train_loader,config,device,optimizer)
        row={'stage':stage,'epoch':epoch,**{'train_'+k:v for k,v in train_metrics.items()}}
        if val_loader is not None:
            metrics=run_epoch(model,val_loader,config,device)
            row.update({'validation_'+k:v for k,v in metrics.items()})
            if metrics['loss']<best:
                best,best_epoch=metrics['loss'],epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:best_epoch=epoch
        history.append(row)
        print(f'{stage} epoch {epoch}: train loss {train_metrics["loss"]:.4f}' +
              (f' | selection loss {metrics["loss"]:.4f}' if val_loader is not None else ''))
    if best_state is not None:model.load_state_dict(best_state)
    return model,best_epoch


def transformations(mode):
    return [(k,f) for k in range(4) for f in (False,True)] if mode in ('d4','d4_scale') else [(0,False)]


def predict_views(model,bases,indices,labels,residue,config,device,mode):
    """Use the same views on all test images, regardless of their labels."""
    model.eval()
    views,names=[],[]
    scales=config.tta_scales if mode in ('scale','d4_scale') else (1.,)
    with torch.no_grad():
        for scale in scales:
            data=loader(bases[scale],indices,labels,residue,config,0)
            pieces=[[] for _ in transformations(mode)]
            for images,_ in data:
                images=images.to(device)
                for j,(k,flip) in enumerate(transformations(mode)):
                    image=torch.rot90(images,k,[-2,-1])
                    if flip:image=torch.flip(image,[-1])
                    output=model(image.contiguous())
                    output=output.softmax(1) if config.task==Task.BIN_CLASSIFICATION else output.reshape(-1)
                    pieces[j].append(output.cpu().numpy())
            for (k,flip),parts in zip(transformations(mode),pieces):
                views.append(np.concatenate(parts))
                names.append(f'scale_{scale}_rot_{90*k}_flip_{int(flip)}')
    return np.stack(views),names


def plot_results(rows,config,out,prefix):
    y=np.array([r['target'] for r in rows]);p=np.array([r['prediction'] for r in rows])
    if config.task==Task.BIN_CLASSIFICATION:
        cm=confusion_matrix(y,p,labels=[0,1,2])
        write_csv(out/f'{prefix}_confusion.csv',[{'actual':name,**dict(zip(CLASS_NAMES,row))} for name,row in zip(CLASS_NAMES,cm)])
        fig,ax=plt.subplots(figsize=(5,4))
        ax.imshow(cm,cmap='Blues')
        for i in range(3):
            for j in range(3):
                pct=cm[i,j]/cm[i].sum() if cm[i].sum() else 0
                ax.text(j,i,f'{cm[i,j]}\n{pct:.1%}',ha='center',va='center')
        ax.set(xticks=range(3),xticklabels=CLASS_NAMES,yticks=range(3),yticklabels=CLASS_NAMES,
               xlabel='Predicted class',ylabel='Measured class',title=prefix)
    else:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].scatter(y,p,s=15);lo=min(y.min(),p.min());hi=max(y.max(),p.max())
        axes[0].plot([lo,hi],[lo,hi],'k--');axes[0].set(xlabel='Measured mg/cm²',ylabel='Predicted mg/cm²')
        axes[1].scatter(p,y-p,s=15);axes[1].axhline(0,color='black',ls='--')
        axes[1].set(xlabel='Predicted mg/cm²',ylabel='Measured - predicted mg/cm²')
        fig.suptitle(prefix)
    fig.tight_layout();fig.savefig(out/f'{prefix}.png',dpi=180);plt.close(fig)


def row_metrics(rows,config):
    y=[r['target'] for r in rows];p=[r['prediction'] for r in rows]
    if config.task==Task.BIN_CLASSIFICATION:
        metrics=classification_metrics(y,p,[[r['probability_'+c] for c in CLASS_NAMES] for r in rows])
        metrics['majority_baseline_accuracy']=float(np.mean(np.array(y)==[r['baseline_prediction'] for r in rows]))
    else:
        metrics=regression_metrics(y,p)
        metrics['train_mean_baseline_mae']=float(np.mean(np.abs(np.array(y)-[r['baseline_prediction'] for r in rows])))
        metrics['train_median_baseline_mae']=float(np.mean(np.abs(np.array(y)-[r['baseline_median'] for r in rows])))
    return metrics


def checkpoint_path(config,seed,fold):
    return config.source_checkpoint_dir/f'{config.task.value}_{config.backbone.value}_seed_{seed}_fold_{fold}.pt'


def load_checkpoint(path):
    # Legacy checkpoints contain NumPy objects.
    return torch.load(path,map_location='cpu',weights_only=False)


def check_saved_config(checkpoint,config):
    saved=checkpoint.get('config',{})
    for key in ('task','backbone','csv_path'):
        if key not in saved:raise ValueError(f'Checkpoint missing {key}; cannot establish provenance.')
        wanted=getattr(config,key);wanted=wanted.value if isinstance(wanted,Enum) else wanted
        if key!='csv_path' and saved[key]!=wanted:raise ValueError(f'Checkpoint {key} mismatch.')
    saved_pre=saved.get('preprocessing','auto')
    if saved_pre=='auto':saved_pre='dinov3_256' if saved['backbone']=='dinov3' else 'legacy_224'
    if saved_pre!=preprocessing_name(config):raise ValueError('Preprocessing differs from saved checkpoint.')
    if tuple(saved.get('class_names',CLASS_NAMES))!=CLASS_NAMES:raise ValueError('Checkpoint class order differs.')
    if saved.get('bin_policy','csv')!=config.bin_policy:raise ValueError('Checkpoint bin policy differs.')


def json_config(config):
    return {k:(v.value if isinstance(v,Enum) else str(v) if isinstance(v,Path) else v) for k,v in asdict(config).items()}


def main(config=CONFIG):
    if config.epochs<1 or config.folds<2 or not config.random_seeds:raise ValueError('Invalid epochs/folds/seeds.')
    if len(set(config.random_seeds))!=len(config.random_seeds):raise ValueError('Duplicate seeds.')
    if 'single' not in config.inference_modes:raise ValueError('Include single inference for a paired reference.')
    if len(set(config.inference_modes))!=len(config.inference_modes):raise ValueError('Duplicate inference modes.')
    if config.class_names!=CLASS_NAMES:raise ValueError('This protocol expects low, medium, high in that order.')
    if any(m not in ('single','d4','scale','d4_scale') for m in config.inference_modes):raise ValueError('Unknown inference mode.')
    if config.augmentation not in ('none','hflip','d4','medium_d4'):raise ValueError('Unknown augmentation.')
    if config.preprocessing not in ('auto','dinov3_256','legacy_224'):raise ValueError('Unknown preprocessing.')
    if config.selection not in ('inner_refit','fixed','legacy_outer'):raise ValueError('Unknown selection mode.')
    if len(set(config.tta_scales))!=len(config.tta_scales) or any(s<=0 for s in config.tta_scales):raise ValueError('Invalid TTA scales.')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base=agai_correct_v3(config.csv_path,transform=make_transform(config))
    metadata=base.icp_data.reset_index(drop=True).copy()
    if len(metadata)!=len(base):raise ValueError('Dataset and metadata lengths differ.')
    required=[config.sample_id_column,config.residue_column,config.residue_bin_column]
    if config.group_column:required.append(config.group_column)
    if metadata[required].isna().any().any():raise ValueError('Missing required metadata; resolve exclusions first.')
    ids=metadata[config.sample_id_column].astype(str).to_numpy()
    if len(set(ids))!=len(ids):raise ValueError('Sample IDs must identify distinct images; configure a unique ID column.')
    residue=pd.to_numeric(metadata[config.residue_column],errors='raise').to_numpy(dtype=float)
    csv_labels=encode_bins(metadata[config.residue_bin_column])
    if not np.isfinite(residue).all():raise ValueError('Nonfinite residues.')
    for i in (0,len(base)-1):
        _,base_target=base[i]
        if not np.isclose(float(base_target),residue[i],rtol=1e-5,atol=1e-7):
            raise ValueError('Configured residue_column disagrees with the existing dataset loader target.')
    groups=None if config.group_column is None else metadata[config.group_column].astype(str).to_numpy()
    bases={1.:base}
    if any(m in ('scale','d4_scale') for m in config.inference_modes):
        for scale in config.tta_scales:
            if scale==1.:continue
            scaled=agai_correct_v3(config.csv_path,transform=make_transform(config,scale))
            if scaled.icp_data[config.sample_id_column].astype(str).tolist()!=ids.tolist():raise ValueError('Scaled dataset order changed.')
            bases[scale]=scaled
    fingerprint=hashlib.sha256(Path(config.csv_path).read_bytes()).hexdigest()
    run_name=config.run_name or f'{config.task.value}_{config.backbone.value}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
    if Path(run_name).name!=run_name or run_name in ('.','..'):raise ValueError('run_name must be a folder name.')
    out=config.output_root/run_name;out.mkdir(parents=True,exist_ok=False)
    manifest={'config':json_config(config),'resolved_preprocessing':preprocessing_name(config),'csv_sha256':fingerprint,
              'sample_ids_in_order':ids.tolist(),'n':len(ids),'torch_version':torch.__version__,'device':str(device),'precision':'float32',
              'validation_note':'Outer test unused for epoch selection in inner_refit/fixed. Legacy checkpoints retain selection bias.'}
    if config.backbone==Backbone.DINOV3:
        revision=subprocess.run(['git','-C',config.dinov3_repo_dir,'rev-parse','HEAD'],capture_output=True,text=True)
        manifest['dinov3_commit']=revision.stdout.strip() if revision.returncode==0 else None
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    metadata.to_csv(out/'dataset_manifest.csv',index=False)
    print(f'{config.task.value} | {config.backbone.value} | {config.selection} | {preprocessing_name(config)}')
    print(f'{len(ids)} images | output: {out}')
    all_rows=[];fold_metrics=[];distributions=[];split_rows=[];threshold_rows=[];all_views=[]
    for seed in config.random_seeds:
        if config.evaluate_only:
            splits=[]
            for fold in range(1,config.folds+1):
                saved=load_checkpoint(checkpoint_path(config,seed,fold));check_saved_config(saved,config)
                test=np.asarray(saved['val_indices'],dtype=int)
                splits.append((np.setdiff1d(np.arange(len(ids)),test),test))
            check_partition(splits,len(ids),groups)
        else:splits=make_outer_splits(csv_labels,config.folds,seed,groups)
        for fold,(train_idx,test_idx) in enumerate(splits,1):
            print(f'Seed {seed} | fold {fold}/{config.folds}')
            fold_seed=seed+fold-1  # Match the earlier script's zero-based fold RNG offset.
            labels,edges=assign_labels(residue,csv_labels,train_idx,config.bin_policy,config.thresholds,config.quantiles)
            if len(np.unique(labels[train_idx]))<3:raise ValueError('An outer training class is empty.')
            threshold_rows.append({'seed':seed,'fold':fold,'policy':config.bin_policy,
                                   'lower':None if edges is None else edges[0],'upper':None if edges is None else edges[1]})
            for role,idx in [('outer_train',train_idx),('outer_test',test_idx)]:
                distributions.extend(distribution_rows(idx,labels,residue,seed,fold,role))
                split_rows.extend({'seed':seed,'fold':fold,'role':role,'sample_id':ids[i],'dataset_index':i,
                                   'group':None if groups is None else groups[i]} for i in idx)
            if config.evaluate_only:
                path=checkpoint_path(config,seed,fold);saved=load_checkpoint(path)
                if saved.get('csv_sha256') and saved['csv_sha256']!=fingerprint:raise ValueError('CSV changed since checkpoint creation.')
                if saved.get('sample_ids') is not None and saved['sample_ids']!=ids.tolist():raise ValueError('Checkpoint sample order changed.')
                expected=labels[test_idx] if config.task==Task.BIN_CLASSIFICATION else residue[test_idx]
                if not np.allclose(saved['val_targets'],expected,rtol=1e-5,atol=1e-7):raise ValueError('Saved targets do not match current sample order/labels.')
                model=build_model(config,device);model.load_state_dict(saved['model_state_dict'])
                epoch=int(saved['epoch'])+1
                selection_used=saved.get('config',{}).get('selection','legacy_outer')
            else:
                history=[]
                if config.selection=='inner_refit':
                    inner_train,inner_val=make_inner_split(train_idx,labels,fold_seed,groups,config.inner_fraction,config.inner_group_folds)
                    for role,idx in [('inner_train',inner_train),('inner_validation',inner_val)]:
                        distributions.extend(distribution_rows(idx,labels,residue,seed,fold,role))
                        split_rows.extend({'seed':seed,'fold':fold,'role':role,'sample_id':ids[i],'dataset_index':i,
                                           'group':None if groups is None else groups[i]} for i in idx)
                    selection_model,epoch=fit_model(inner_train,inner_val,base,labels,residue,config,device,fold_seed,'inner_selection',history)
                    del selection_model
                    model,_=fit_model(train_idx,None,base,labels,residue,config,device,fold_seed,'outer_refit',history,epoch)
                else:
                    val=test_idx if config.selection=='legacy_outer' else None
                    model,epoch=fit_model(train_idx,val,base,labels,residue,config,device,fold_seed,config.selection,history)
                write_csv(out/f'history_seed_{seed}_fold_{fold}.csv',history)
                selection_used=config.selection
            fold_rows=[];single=None
            modes=('single',)+tuple(m for m in config.inference_modes if m!='single')
            for mode in modes:
                values,names=predict_views(model,bases,test_idx,labels,residue,config,device,mode)
                aggregate,instability=aggregate_views(values,config.task.value)
                if mode=='single':methods=(('single','mean'),)
                elif config.task==Task.BIN_CLASSIFICATION:methods=((mode+'_mean','mean'),(mode+'_vote','vote'))
                else:methods=((mode+'_mean','mean'),)
                for condition,method in methods:
                    for j,i in enumerate(test_idx):
                        target=labels[i] if config.task==Task.BIN_CLASSIFICATION else residue[i]
                        row={'task':config.task.value,'backbone':config.backbone.value,'seed':seed,'fold':fold,'best_epoch':epoch,
                             'selection':selection_used,'inference':condition,'dataset_index':int(i),'sample_id':ids[i],
                             'group':None if groups is None else groups[i],'residue_mg_cm2':residue[i],
                             'target':int(target) if config.task==Task.BIN_CLASSIFICATION else float(target),
                             'prediction':float(aggregate[method][j]),'view_count':len(names),'view_instability':float(instability[j])}
                        if config.task==Task.BIN_CLASSIFICATION:
                            row['prediction']=int(row['prediction'])
                            row.update({'target_label':CLASS_NAMES[int(target)],'prediction_label':CLASS_NAMES[row['prediction']],
                                        'baseline_prediction':int(np.bincount(labels[train_idx],minlength=3).argmax())})
                            row.update({'probability_'+c:float(aggregate['probabilities'][j,k]) for k,c in enumerate(CLASS_NAMES)})
                        else:row.update({'residual':float(target-row['prediction']),'baseline_prediction':float(residue[train_idx].mean()),
                                         'baseline_median':float(np.median(residue[train_idx]))})
                        fold_rows.append(row)
                for v,name in enumerate(names):
                    for j,i in enumerate(test_idx):
                        row={'seed':seed,'fold':fold,'sample_id':ids[i],'mode':mode,'view':name}
                        if config.task==Task.BIN_CLASSIFICATION:row.update({'probability_'+c:float(values[v,j,k]) for k,c in enumerate(CLASS_NAMES)})
                        else:row['prediction']=float(values[v,j])
                        all_views.append(row)
                if mode=='single':
                    single=values[0]
                    if config.evaluate_only:
                        reference=saved.get('val_probabilities') if config.task==Task.BIN_CLASSIFICATION else saved['val_predictions']
                        current=single if reference is not None else single.argmax(1)
                        if reference is None:reference=saved['val_predictions']
                        max_diff=float(np.max(np.abs(np.asarray(reference)-current)))
                        if not np.allclose(reference,current,rtol=1e-4,atol=1e-5):
                            raise ValueError(f'Single-view replay differs (max {max_diff:g}); check model, transforms and data before comparing TTA.')
                        print(f'Checkpoint single-view replay matched (max difference {max_diff:g}).')
                    else:
                        # Preserve the completed fit before potentially expensive TTA.
                        torch.save({'epoch':epoch-1,'fold':fold-1,'seed':seed,'config':json_config(config),
                                    'model_state_dict':model.state_dict(),'csv_sha256':fingerprint,'sample_ids':ids.tolist(),
                                    'val_indices':test_idx,'val_targets':labels[test_idx] if config.task==Task.BIN_CLASSIFICATION else residue[test_idx],
                                    'val_predictions':single.argmax(1) if config.task==Task.BIN_CLASSIFICATION else single,
                                    'val_probabilities':single if config.task==Task.BIN_CLASSIFICATION else None},
                                   out/f'{config.task.value}_{config.backbone.value}_seed_{seed}_fold_{fold}.pt')
            for condition in dict.fromkeys(r['inference'] for r in fold_rows):
                rows=[r for r in fold_rows if r['inference']==condition]
                metrics=row_metrics(rows,config)
                fold_metrics.append({'seed':seed,'fold':fold,'inference':condition,'n':len(rows),**metrics})
                print(condition,metrics)
                plot_results(rows,config,out,f'seed_{seed}_fold_{fold}_{condition}')
            all_rows.extend(fold_rows)
            write_csv(out/'oof_predictions.csv',all_rows);write_csv(out/'fold_metrics.csv',fold_metrics)
            write_csv(out/'class_distributions.csv',distributions);write_csv(out/'splits.csv',split_rows)
            write_csv(out/'thresholds.csv',threshold_rows);write_csv(out/'view_predictions.csv',all_views)
            del model
    summary=[]
    for seed in config.random_seeds:
        for condition in dict.fromkeys(r['inference'] for r in all_rows):
            rows=[r for r in all_rows if r['seed']==seed and r['inference']==condition]
            if len(rows)!=len(ids):raise ValueError('Incomplete OOF coverage.')
            summary.append({'seed':seed,'inference':condition,'n':len(rows),**row_metrics(rows,config)})
            plot_results(rows,config,out,f'seed_{seed}_pooled_{condition}')
    write_csv(out/'seed_metrics.csv',summary)
    if len(config.random_seeds)>1:
        numeric=pd.DataFrame(summary).drop(columns=['n']).groupby('inference').agg(['mean','std']).drop(columns='seed',level=0)
        numeric.columns=['_'.join(c) for c in numeric.columns]
        numeric.to_csv(out/'seed_stability.csv')
    comparison=[]
    for seed in config.random_seeds:
        reference=next(r for r in summary if r['seed']==seed and r['inference']=='single')
        for row in summary:
            if row['seed']!=seed or row['inference']=='single':continue
            comparison.append({'seed':seed,'inference':row['inference'],**{'delta_'+k:row[k]-reference[k] for k in reference if k not in ('seed','inference','n')}})
    write_csv(out/'tta_deltas.csv',comparison)
    print(f'Completed: {out}/seed_metrics.csv and tta_deltas.csv')


def parse_config():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=[t.value for t in Task]);p.add_argument('--backbone',choices=[b.value for b in Backbone])
    p.add_argument('--csv-path');p.add_argument('--group-column');p.add_argument('--residue-column');p.add_argument('--sample-id-column')
    p.add_argument('--preprocessing',choices=['auto','dinov3_256','legacy_224'])
    p.add_argument('--augmentation',choices=['none','hflip','d4','medium_d4'])
    p.add_argument('--bin-policy',choices=['csv','fixed','train_quantile'])
    p.add_argument('--thresholds',nargs=2,type=float);p.add_argument('--quantiles',nargs=2,type=float)
    p.add_argument('--selection',choices=['inner_refit','fixed','legacy_outer'])
    p.add_argument('--seeds',nargs='+',type=int);p.add_argument('--epochs',type=int);p.add_argument('--folds',type=int)
    p.add_argument('--batch-size',type=int);p.add_argument('--learning-rate',type=float);p.add_argument('--num-workers',type=int)
    p.add_argument('--inference-modes',nargs='+',choices=['single','d4','scale','d4_scale'])
    p.add_argument('--tta-scales',nargs='+',type=float)
    p.add_argument('--evaluate-only',action='store_true',default=None)
    p.add_argument('--checkpoint-dir',dest='source_checkpoint_dir',type=Path)
    p.add_argument('--output-root',type=Path);p.add_argument('--run-name')
    p.add_argument('--dinov3-repo-dir');p.add_argument('--dinov3-weights-path')
    p.add_argument('--legacy-resnet-bn',action='store_true',help='Legacy BatchNorm updates for historical replication.')
    args={k:v for k,v in vars(p.parse_args()).items() if v is not None}
    if args.pop('legacy_resnet_bn',False):args['freeze_resnet_bn']=False
    if 'seeds' in args:args['random_seeds']=tuple(args.pop('seeds'))
    for key in ('thresholds','quantiles','inference_modes','tta_scales'):
        if key in args:args[key]=tuple(args[key])
    if 'task' in args:args['task']=Task(args['task'])
    if 'backbone' in args:args['backbone']=Backbone(args['backbone'])
    return replace(CONFIG,**args)


if __name__=='__main__':
    main(parse_config())