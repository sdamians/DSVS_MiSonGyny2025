import time
import torch as t
import torch.nn as nn
import numpy as np
import gc
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from utils import format_time
from utils import get_metrics

class Learner:
  def __init__(self, classifier, 
               class_weights: Tensor, 
               optimizer_params: dict, 
               criterion_params: dict, 
               scheduler_params: dict, 
               device="cpu"):
    
    self.model = classifier

    self.optimizer = t.optim.Adam(self.model.parameters(), **optimizer_params)

    if "reduction" in criterion_params:
      self.criterion = nn.CrossEntropyLoss(weight=class_weights, **criterion_params)
    else:
      if criterion_params["alpha"] is not None:
        criterion_params["alpha"] = t.tensor(criterion_params["alpha"]).to(device)
      self.criterion = FocalLoss(**criterion_params)

    self.device = device

    self.scheduler_params = scheduler_params

  def train(self, trainset: DataLoader, valset: DataLoader, n_epochs: int, gradient_accumulator_size: int=2):
    t_gral = time.time()

    max_step_t = len(trainset)
    total_training_steps = (len(trainset.dataset) // (trainset.batch_size * gradient_accumulator_size)) * n_epochs

    # scheduler = t.optim.lr_scheduler.PolynomialLR(self.optimizer, **self.scheduler_params)
    
    scheduler = get_cosine_schedule_with_warmup(
        self.optimizer,
        num_warmup_steps=int(0.1 * total_training_steps),  # Warmup del 10%
        num_training_steps=(total_training_steps),
        **self.scheduler_params  # Opcional: media onda de coseno (default)
    )

    scaler = t.amp.GradScaler(device=self.device)

    # Training mode.
    self.model.train()
    self.model.zero_grad()
  
    for epoch in range(n_epochs):
      t0 = time.time() # We save the start time to see how long it takes.
      epoch_loss = []  # We reset the loss value for each epoch.
      
      with tqdm(total=len(trainset), desc=f'Epoch {epoch + 1}/{n_epochs}', dynamic_ncols=True) as pbar:
        for step, batch in enumerate(trainset):
          batch_loss = 0

          input_ids = batch["input_ids"].to(self.device)
          attention_mask = batch["attention_mask"].to(self.device)
          labels = batch["labels"].to(self.device)

          if "pad_len" in batch:
            num_verses = batch["pad_len"]

          with t.amp.autocast(device_type=self.device):

            # Propagation forward in the layers
            if "pad_len" in batch:
              outputs = self.model(input_ids, attention_mask=attention_mask, num_verses=num_verses)
            else:
              outputs = self.model(input_ids, attention_mask=attention_mask)

            # We calculate the loss of the present minibatch
            loss = self.criterion(outputs["logits"], labels) #outputs[0]
            batch_loss += loss.item()
            pbar.set_postfix({ "loss": loss.item() })
            pbar.update(1)

          # Backpropagation
          scaler.scale(loss).backward()
          #loss.backward()

          # So we can implement gradient accumulator technique
          if (step > 0 and step % gradient_accumulator_size == 0) or (step == max_step_t - 1):

            #(this prevents the gradient from becoming explosive)
            t.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # We update the weights and bias according to the optimizer
            scaler.step(self.optimizer) #self.optimizer.step()
            scaler.update()
            #Update learning rate each end of epoch
            scheduler.step()

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

          # if (step % ( max_step_t // 5 ) == 0) or (step == max_step_t - 1):
          #   print(f"Batch {step}/{max_step_t} avg loss: {np.sum(epoch_loss) / (step+1):.5f}")
        
          epoch_loss.append(batch_loss)

      # We calculate the average loss in the current epoch of the training set
      print(f"\n\tAverage training loss: {np.sum(epoch_loss)/max_step_t:.4f}")
      print(f"\tTraining epoch {epoch + 1} took: {format_time(time.time() - t0)}")

      if valset is not None:
        print("\n\tValidation step:")
        # self.test(trainset, "trainset metrics:")
        self.test(valset, "valset metrics:")

    print(f"\nTraining complete. It took: {format_time(time.time() - t_gral)}")

  def test(self, testset: DataLoader, msg: str):
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
        labels = batch["labels"].to(self.device) if "labels" in batch else None
        if "pad_len" in batch:
          num_verses = batch["pad_len"]

        # [ batch  ]
        if "pad_len" in batch:
          outputs = self.model(input_ids, attention_mask=attention_mask, num_verses=num_verses)
        else:
          outputs = self.model(input_ids, attention_mask=attention_mask)

        probs = t.softmax(outputs["logits"], dim=-1) # [batch n_classes]
        preds = t.argmax(probs, dim=1).unsqueeze(-1) # [batch 1]

        #probs = t.gather(probs, dim=-1, index=preds)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        if labels is not None:
          all_labels.extend(labels.cpu().numpy())
          loss = self.criterion(outputs["logits"], labels)
          eval_loss += loss

    # We show the final metric scores
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds).flatten()
    
    if labels is not None:
      all_labels = np.array(all_labels).flatten()

      metrics = get_metrics(all_labels, all_preds, all_probs[:, 1], promedio='macro')

      print(msg)
      for key in metrics:
        print(f"\t{key}: {metrics[key]}")
      print(f"\tEvalLoss: {eval_loss / len(testset):.4f}")

      print(f"\tValidation took: {format_time(time.time() - t0)}")

      return all_preds, all_probs, all_labels, eval_loss
    
    return all_preds, all_probs
  

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        """
        alpha: Tensor de pesos para cada clase, shape (num_classes,)
        gamma: Focusing parameter
        """
        super().__init__()
        if alpha is not None and not isinstance(alpha, Tensor):
            raise ValueError("alpha must be a Tensor")
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = t.exp(-ce_loss)

        if self.alpha is not None:
            # Selecciona el alpha correspondiente a cada target
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()