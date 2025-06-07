import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch
torch.multiprocessing.set_start_method('spawn', force=True)  # Add this first
from fastai.vision.all import *
from fastai.metrics import accuracy
from PIL import ImageFile
from pathlib import Path

plt.style.use('ggplot')  # Using a different valid style
ImageFile.LOAD_TRUNCATED_IMAGES = True
classes = ['Boletus','Entoloma','Russula','Suillus','Lactarius','Amanita','Agaricus','Hygrocybe','Cortinarius']
bs = 128  # Reduced batch size if you're having memory issues
path = Path('Kaggle')
path.mkdir(parents=True, exist_ok=True)
set_seed(42)

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.35, seed=42),
    get_y=parent_label,
    item_tfms=Resize(224),
    batch_tfms=aug_transforms() + [Normalize.from_stats(*imagenet_stats)]
).dataloaders(
    path,
    bs=bs,
    num_workers=0 if os.name == 'nt' else min(4, os.cpu_count()),
    pin_memory=torch.cuda.is_available()
)


dls.show_batch(max_n=9, figsize=(7,8))
print(dls.vocab, len(dls.train_ds), len(dls.valid_ds))

learn = vision_learner(
    dls, 
    resnet50, 
    metrics=accuracy,
    cbs=[MixUp()] 
).to_fp16()  

learn.fit_one_cycle(8, wd=0.9)

learn.unfreeze()
learn.lr_find()

# Plot learning rate finder results
# plt.figure()
# learn.recorder.plot_lr_find()
# plt.show()


learn.fit_one_cycle(4, lr_max=slice(1e-5, 2e-3), pct_start=0.8, wd=0.9)
# Save the model
model_path=Path(f'model_1.pkl')
model_dictionary=Path(f'model_1.pth')
learn.export(model_path)

# Save just the weights (alternative)
torch.save(learn.model.state_dict(), model_dictionary)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
plt.show()
test_path = 'KaggleTest'

test_dl = dls.test_dl(get_image_files(test_path), with_labels=True)
preds, y = learn.get_preds(dl=test_dl)
print("Accuracy on test set:", accuracy(preds, y).item())
