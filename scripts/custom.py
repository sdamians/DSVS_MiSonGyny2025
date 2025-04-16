import time
import torch as t
import torch.nn as nn
import numpy as np
import gc

from torch import Tensor
from torch.utils.data import DataLoader
from textclassifier import TextClassifier
from datasets import Dataset
from tqdm import tqdm

class Learner:
  def __init__(self, model_path, class_weights: Tensor, device="cpu"):
    self.model = TextClassifier(model_path, 
                                num_labels=2, 
                                get_att=False, 
                                get_hs=False, 
                                dropout=0.1)

    self.optimizer = t.optim.Adam(self.model.parameters()
                                ,lr=1e-5
                                ,eps=1e-8
                                ,weight_decay=0.01
                                ,betas=(0.9, 0.999) # 0.9, 0.999
                                ,amsgrad=False)

    self.criterion = nn.CrossEntropyLoss(weight=class_weights # torch.tensor([5.8, 0.43, 2]).to(device)
                                        ,reduction='sum' #sum, mean
                                        ,label_smoothing=0)

    self.device = device

  def train(self, trainset: DataLoader, valset: DataLoader, n_epochs: int, gradient_accumulator_size: int=2):
    max_step_t = len(trainset)
    max_step_v = len(valset)

    scheduler = t.optim.lr_scheduler.PolynomialLR(self.optimizer
                                                      ,total_iters=n_epochs
                                                      ,power=1.0)

    total_loss = []
    total_lr = []

    for epoch in tqdm(range(n_epochs), desc="Epoch: "):
      # We save the start time to see how long it takes.
      t0 = time.time()

      # We reset the loss value for each epoch.
      epoch_loss = []

      # Training mode.
      self.model.train()
      self.model.zero_grad()

      for step, batch in enumerate(trainset):
        batch_loss = 0
        b_input_ids, b_input_mask, b_labels = tuple(t.to(self.device) for t in batch)

        # Propagation forward in the layers
        outputs = self.model(b_input_ids,
                        attention_mask=b_input_mask)

        # We calculate the loss of the present minibatch
        loss = self.criterion(outputs, b_labels) #outputs[0]
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

        b_input_ids.to("cpu")
        b_input_mask.to("cpu")
        b_labels.to("cpu")

        del b_input_ids
        del b_input_mask
        del b_labels
        
        t.cuda.empty_cache()
        gc.collect()

        if (step % (step  // 10) == 0) or (step == max_step_t - 1):
          print(f"Batch {step}/{max_step_t} avg loss: {np.sum(epoch_loss) / (step+1):.5f}")

      #Update learning rate each end of epoch
      scheduler.step()
      total_lr.append(scheduler.get_last_lr())
      total_loss.append(np.sum(epoch_loss)/max_step_t)

      # We calculate the average loss in the current epoch of the training set
      print(f"\n\tAverage training loss: {np.sum(epoch_loss)/max_step_t):.5f}")
      print(f"\tTraining epoch took: {format_time(time.time() - t0)}")

      print("\n\tValidation")
      curr_score, curr_eval_loss = self.validation(val_dataloader)

    # Display learning rate and loss charts
    """
    epoch_axis = np.arange(0, n_epochs) + 1
    plt.plot(epoch_axis, total_loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.xticks(epoch_axis)
    plt.show()
    """
    """
    print("\n\n")

    plt.plot(epoch_axis, total_lr)
    plt.xlabel("epoch")
    plt.ylabel("learning rate")
    plt.xticks(epoch_axis)
    plt.show()
    """
    print("\nTraining complete")

  def validate(self, val_dataloader):
    t0 = time.time()

    # We put the model in validation mode
    self.model.eval()

    # We declare variables
    eval_loss = 0
    eval_metric = 0
    all_logits = []
    all_labels = []

    # By minibatches
    for step, batch in enumerate(val_dataloader):
      b_input_ids, b_input_mask, b_labels = tuple(t.to(DEVICE) for t in batch)

      with torch.no_grad():
        # We generate the predictions of the model
        outputs = self.model(b_input_ids,
                            attention_mask=b_input_mask)

        loss = self.criterion(outputs, b_labels)

        # ...we extract them
        logits = torch.argmax(outputs, dim=1).detach().cpu()
        b_labels = b_labels.to('cpu')

        # Saving logits and labels. They will be useful for the confusion matrix.
        #predict_labels = np.argmax(logits, axis=1).flatten()
        all_logits.extend(logits.tolist())
        all_labels.extend(b_labels.tolist())

        eval_loss += loss

    # We calculate the F1 score of this batch
    scores = MulticlassF1Score(num_classes=2, average=None)(torch.tensor(all_logits), torch.tensor(all_labels))
    score = BinaryF1Score()(torch.tensor(all_logits), torch.tensor(all_labels))
    # We show the final accuracy for this epoch
    print(f"\n\tF1Scores: {scores.tolist()}")
    print(f"\n\tEvalLoss: {eval_loss}")
    print(f"\tValidation took: {format_time(time.time() - t0)}")
    return scores, eval_loss

  def testing(self, dataloader):
    preds = []
    labs = []

    with torch.no_grad():
      for step, batch in enumerate(dataloader):
        test_inputs, test_masks, b_labels = tuple(t.to(DEVICE) for t in batch)

        outputs = self.model(test_inputs,
                             attention_mask=test_masks)

        logits = torch.argmax(outputs, dim=1).detach().cpu()
        b_labels = b_labels.to('cpu').tolist()

        preds.extend([x.item() for x in logits])
        labs.extend(b_labels)

    preds = [ "P" if pred == 0 else "NP" for pred in preds ]
    labs = [ str(l) + "_Track3" for l in labs ]

    dataframe_res = { "sub_id": labs, "label": preds }
    dataframe_res = pd.DataFrame.from_dict(dataframe_res)

    return dataframe_res

  def metrics_testing(self, dataloader):
    preds = []
    labs = []

    with torch.no_grad():
      for step, batch in enumerate(dataloader):
        test_inputs, test_masks, b_labels = tuple(t.to(DEVICE) for t in batch)

        outputs = self.model(test_inputs,
                             attention_mask=test_masks)

        logits = torch.argmax(outputs, dim=1).detach().cpu()
        b_labels = b_labels.to('cpu').tolist()

        preds.extend([x.item() for x in logits])
        labs.extend(b_labels)

    return preds, labs
