import time
import torch as t
import torch.nn as nn
import numpy as np
import gc

from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import format_time
from utils import get_metrics

class Learner:
  def __init__(self, classifier, class_weights: Tensor, optimizer_params: dict, criterion_params: dict, scheduler_params: dict,  device="cpu"):
    self.model = classifier

    self.optimizer = t.optim.Adam(self.model.parameters(), **optimizer_params)

    self.criterion = nn.CrossEntropyLoss(weight=class_weights, **criterion_params)

    self.device = device

    self.scheduler_params = scheduler_params

  def train(self, trainset: DataLoader, valset: DataLoader, n_epochs: int, gradient_accumulator_size: int=2):
    batch_size = trainset.batch_size
    max_step_t = len(trainset)

    scheduler = t.optim.lr_scheduler.PolynomialLR(self.optimizer, **self.scheduler_params)

    total_loss = []
    total_lr = []

    for epoch in range(n_epochs):
      # We save the start time to see how long it takes.
      t0 = time.time()

      # We reset the loss value for each epoch.
      epoch_loss = []

      # Training mode.
      self.model.train()
      self.model.zero_grad()

      for step, batch in tqdm(enumerate(trainset)):
        batch_loss = 0

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        num_verses = batch["pad_len"]

        # Propagation forward in the layers
        outputs = self.model(input_ids,
                        attention_mask=attention_mask,
                             num_verses=num_verses)

        # We calculate the loss of the present minibatch
        loss = self.criterion(outputs, labels) #outputs[0]
        batch_loss += loss.item()
        epoch_loss.append( loss.item() )

        # Backpropagation
        loss.backward()

        # So we can implement gradient accumulator technique
        if (step > 0 and step % gradient_accumulator_size == 0) or (step == max_step_t - 1):
          
          #(this prevents the gradient from becoming explosive)
          t.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

          # We update the weights and bias according to the optimizer
          self.optimizer.step()

          # We clean the gradients for the accumulator batch
          self.model.zero_grad()

        input_ids.to("cpu")
        attention_mask.to("cpu")
        labels.to("cpu")

        del input_ids
        del attention_mask
        del labels
        
        t.cuda.empty_cache()
        gc.collect()

        if (step % 50 == 0) or (step == max_step_t - 1):
          print(f"Batch {step}/{max_step_t} avg loss: {np.sum(epoch_loss) / (step+1):.5f}")

      #Update learning rate each end of epoch
      scheduler.step()
      total_lr.append(scheduler.get_last_lr())
      total_loss.append(np.sum(epoch_loss)/max_step_t)

      # We calculate the average loss in the current epoch of the training set
      print(f"\n\tAverage training loss: {np.sum(epoch_loss)/max_step_t:.5f}")
      print(f"\tTraining epoch {epoch + 1} took: {format_time(time.time() - t0)}")

      print("\n\tValidation step:")
      self.test(valset)

    print("\nTraining complete")

  def test(self, testset: DataLoader):
    t0 = time.time()

    # We put the model in validation mode
    self.model.eval()

    # We declare variables
    eval_loss = 0
    all_probs  = []
    all_preds  = []
    all_labels = []


    with t.no_grad():
      for batch in testset:
        
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"] if "labels" in batch else None
        num_verses = batch["pad_len"]

        # [ batch  ]
        outputs = self.model(input_ids, attention_mask=attention_mask, num_verses=num_verses)
        probs = t.softmax(outputs, dim=-1)  
        preds = t.argmax(probs, dim=1)
        probs = t.gather(probs, dim=-1, index=preds)

        all_probs.append(probs)
        all_preds.append(preds)

        if labels is not None:
          all_labels.append(labels)
          loss = self.criterion(outputs, labels)
          eval_loss += loss
    
    all_preds = t.cat(all_preds, dim=0)
    all_probs = t.cat(all_probs, dim=0)

    # We show the final accuracy for this epoch
    if labels is not None:
      all_labels = t.cat(all_labels, dim=0)
      metrics = get_metrics(all_labels, all_preds, all_probs, promedio='binary')
      for key in metrics:
        print(f"\n\t{key}: {metrics[key]}")
      print(f"\n\tEvalLoss: {eval_loss}")
    print(f"\tValidation took: {format_time(time.time() - t0)}")
    
    return all_preds, all_probs, all_labels, eval_loss