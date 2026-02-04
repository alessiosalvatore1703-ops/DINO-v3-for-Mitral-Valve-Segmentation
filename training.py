import torch
import numpy as np
import wandb
import os
from collections import defaultdict

THRESH_LEVELS = []

def train_model(
    model,
    train_dataloader,
    val_dataloader,
    epochs,
    optimizer,
    loss_fn,
    metric_fn,
    device,
    patience=np.inf,
    save_path="models",
    log_interval=10,
    scheduler=None,
    ):

    model.to(device)

    best_val_loss = np.inf
    wait = 0
    
    for epoch in range(epochs):
        print(f"Starting epoch {epoch}/{epochs}")
        model.train()
        train_losses = []

        for batch_idx, batch in enumerate(train_dataloader):
            imgs, targets = batch
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if batch_idx % log_interval == 0:
                wandb.log({"train_batch_loss": loss.item()})
                # print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item()}")
        
        train_loss = np.mean(train_losses)
        
        model.eval()
        val_losses = []
        val_metric = []
        iou_at_thresholds = defaultdict(list)
        for batch in val_dataloader:
            imgs, targets = batch
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            with torch.no_grad():
                # standard
                preds = model(imgs) # get raw logits
                loss = loss_fn(preds, targets) # compute loss
                val_losses.append(loss.item())
                val_metric.append(metric_fn(preds, targets).item())

                # compute IOU @ different thresholds
                for threshold in THRESH_LEVELS:
                    preds = model.predict(imgs, threshold=threshold)
                    iou_at_thresholds[threshold].append(metric_fn(preds, targets).item())
                
        
        val_loss = np.mean(val_losses)
        val_metric = np.mean(val_metric)
        for threshold in THRESH_LEVELS:
            wandb.log({f"val_iou_{threshold}": np.mean(iou_at_thresholds[threshold])})
            print(f"Val IOU @ threshold {threshold}: {np.mean(iou_at_thresholds[threshold])}")
        # log to wandb
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_metric": val_metric,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if scheduler is not None:
             scheduler.step(val_loss)

        # early stopping
        if val_loss > best_val_loss:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break
        else:
            wait = 0
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))

        print(f"Epoch {epoch}/{epochs} finished: train loss {train_loss}, val loss {val_loss}, val metric {val_metric}. LR: {optimizer.param_groups[0]['lr']}")
