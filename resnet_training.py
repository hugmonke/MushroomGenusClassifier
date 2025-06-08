import os
import torch
import numpy as np
import pandas as pd
import fastai.vision.all as fai
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import ImageFile
from fastai.metrics import accuracy

# -------------------------------------------------- #
# -------------------------------------------------- #
#                       CONFIG                       #
# -------------------------------------------------- #
# -------------------------------------------------- #
plot=True
save=True
test=True
batch_size = 100 
train_set_path = Path('Kaggle')
torch.multiprocessing.set_start_method('spawn', force=True)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# DATALOADER PARAMS #
resize=300
workers=0
test_split=0.25

dls = fai.DataBlock(
    blocks=(fai.ImageBlock, fai.CategoryBlock),
    get_items=fai.get_image_files,
    splitter=fai.RandomSplitter(valid_pct=test_split),
    get_y=fai.parent_label,
    item_tfms=fai.Resize(resize),
    batch_tfms=fai.aug_transforms() + [fai.Normalize.from_stats(*fai.imagenet_stats)]
).dataloaders(
    source=train_set_path,
    bs=batch_size,
    num_workers=workers if os.name == 'nt' else min(4, os.cpu_count()),
    pin_memory=torch.cuda.is_available()
)

dls.show_batch(max_n=9, figsize=(7,8))
print('Classes: ', dls.vocab, '\nTrain set size: ', len(dls.train_ds), '\nTest set size: ', len(dls.valid_ds))

learner = fai.vision_learner(
    dls, 
    fai.resnet50, 
    metrics=accuracy,
    cbs=[fai.MixUp()] 
).to_fp16()  

learner.fit_one_cycle(n_epoch=8, wd=0.9)
learner.unfreeze()
learner.lr_find()

if plot:
    plt.style.use('ggplot')
    plt.figure()
    learner.recorder.plot_lr_find()
    plt.show()

learner.fit_one_cycle(n_epoch=8, lr_max=slice(1e-5, 2e-3), pct_start=0.8, wd=0.9)

if save:
    model_path=Path(f'model_1.pkl')
    learner.export(model_path)

if test:
    test_path = Path('KaggleTest')
    test_dl = dls.test_dl(fai.get_image_files(test_path), with_labels=True)
    preds, y = learner.get_preds(dl=test_dl)
    print("Accuracy on test set:", accuracy(preds, y).item())

if plot:
    interp = fai.ClassificationInterpretation.from_learner(learner)
    interp.plot_top_losses(9, figsize=(15,11))
    interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
    plt.show()
