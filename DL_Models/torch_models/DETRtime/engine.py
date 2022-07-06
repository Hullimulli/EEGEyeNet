# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
# import os
import sys
from typing import Iterable

import torch

import util.misc as utils
import logging
import numpy as np
import time 

from PIL import Image, ImageDraw, ImageFont

from util.sequence_generator import generate_sequence_targets, generate_sequence_predictions

from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate
from util.box_ops import box_cxw_to_xlxh


import wandb


def draw_windows(img, targets):
    img = (img - img.min()) / (img.max() - img.min())

    img = Image.fromarray(img)

    x0 = x1 = 0
    kbd = ImageFont.truetype("DeepSeg/arial.ttf", 40)
    num_pred = 0

    boxes = targets['boxes']
    boxes = box_cxw_to_xlxh(boxes) * 500

    for idx, box in enumerate(boxes):
        label = targets['labels'][idx]
        if label > 2:
            continue

        x0 = box[0]
        y0 = 0
        x1 = box[1]
        y1 = 129

        draw = ImageDraw.Draw(img)
        draw.rectangle([(x0, y0), (x1, y1)], width=2, outline=0)
        d = ImageDraw.Draw(img)
        if label == 0:
            d.text((int((x1 + x0) / 2 - 9), 40), "F", font=kbd, fill=0)
        elif label == 1:
            d.text((int((x1 + x0) / 2 - 9), 40), "S", font=kbd, fill=0)
        elif label == 2:
            d.text((int((x1 + x0) / 2 - 9), 40), "B", font=kbd, fill=0)

        num_pred += 1
    return np.asarray(img), num_pred


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, timestamps: int, max_norm: float = 0):
    model.train()
    criterion.train()

    # TODO: Logging
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    log_count = 0
    show = False
    if epoch % 10 == 0:
        show = True

    y_true = []
    y_hat = []

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        # print(type(samples[0]))
        #print(samples[0])
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_value = losses.item()
        if log_count % print_freq == 0:
            logging.info(loss_dict)
            logging.info({'loss': losses})
            wandb.log(loss_dict)
            wandb.log({'loss': losses})

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            logging.info("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            logging.info(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict)
        metric_logger.update(class_error=loss_dict['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        log_count += 1

        for batch_idx in range(len(samples)):
            seq_true = generate_sequence_targets(targets[batch_idx], timestamps)

            d = {
                'pred_boxes': outputs['pred_boxes'][batch_idx].detach().cpu(),
                'pred_logits': outputs['pred_logits'][batch_idx].detach().cpu()
            }
            seq_hat = generate_sequence_predictions(d, timestamps)

            y_true.append(seq_true)
            y_hat.append(seq_hat)

        # log sample images and the prediction
        
        
        if show:
            for idx in range(4):
                if idx >= len(outputs['pred_logits']):
                    continue
                images = []

                img = samples[idx].detach().cpu().numpy()
                logits = outputs['pred_logits'][idx]
                labels = np.argmax(logits.detach().cpu().numpy(), axis=-1)
                boxes = outputs['pred_boxes'][idx]
                img_pred, num_pred1 = draw_windows(img, {'boxes': boxes.detach().cpu(), 'labels': labels})

                img_true, num_pred2 = draw_windows(img, targets[idx])
                img_pred = wandb.Image(img_pred, caption=f"Prediction {idx} ({num_pred1})")
                images.append(img_pred)
                img_true = wandb.Image(img_true, caption=f"True Image {idx} ({num_pred2})")
                images.append(img_true)
                
                wandb.log({f'Train image {idx}': images})

            show = False



    print("Averaged stats:", metric_logger)
    logging.info(f"Averaged stats: \n{metric_logger}")

    # log and print classification report
    y_true = np.array(y_true).flatten()
    y_hat = np.array(y_hat).flatten()

    #target_names = ['Fixation', 'Saccade', 'Blink']

    #c_report = classification_report(y_true, y_hat, output_dict=True)
    # for label in ['Fixation', 'Saccade', 'Blink', 'macro avg', 'weighted avg']:
    #    log = {f'{label}_{k}_train':c_report[label][k] for k in c_report[label]}
    #    wandb.log(log)

    # wandb.log({'overall_accuracy_train': c_report['accuracy']})

    #print('TRAINING CLASSIFICATION REPORT:\n', classification_report(y_true, y_hat, digits=4))
    logging.info(f"TRAIN CLASSIFICATION REPORT: \n {classification_report(y_true, y_hat, digits=4)}")


    print('Confusion Matrix train:')
    cm = confusion_matrix(y_true, y_hat)
    logging.info(f"Confusion matrix training: \n{cm}")
    # cm = cm/cm.astype(np.float).sum(axis=1)
    print(tabulate(cm, headers=['fix', 'sacc', 'blink']))
    print()

    # TODO: return important stats
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, epoch, device, output_dir, timestamps: int):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    print_freq = 100

    log_count = 0
    show = False
    if epoch % 10 == 0:
        show = True

    y_true = []
    y_hat = []

    acc_time = 0
    num_batches = 0 
    batch_size = 0 
    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        #print(samples.size())
        batch_size = samples.size()[0]
        start = time.time()
        outputs = model(samples)
        total_time = time.time() - start 
        acc_time += total_time
        #print(f"total time {total_time}")
        num_batches += 1 
        if num_batches > 500:
            break 

        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_value = losses.item()

        if log_count % print_freq == 0:
            logging.info(loss_dict)
            logging.info({'loss': losses})
            wandb.log(loss_dict)
            wandb.log({'loss': losses})

        metric_logger.update(loss=loss_value, **loss_dict)
        metric_logger.update(class_error=loss_dict['class_error'])

        # log sample images and the prediction
        
        show = False
        if show:
            for idx in range(4):
                if idx >= len(outputs['pred_logits']):
                    continue
                images = []

                img = samples[idx].detach().cpu().numpy()
                logits = outputs['pred_logits'][idx]
                labels = np.argmax(logits.detach().cpu().numpy(), axis=-1)
                boxes = outputs['pred_boxes'][idx]
                img_pred, num_pred1 = draw_windows(img, {'boxes': boxes.detach().cpu(), 'labels': labels})
                img_true, num_pred2 = draw_windows(img, targets[idx])
                img_pred = wandb.Image(img_pred, caption=f"Prediction {idx} ({num_pred1})")
                images.append(img_pred)
                img_true = wandb.Image(img_true, caption=f"True Image {idx} ({num_pred2})")
                images.append(img_true)
                
                wandb.log({f'Test image {idx}': images})
            
            show = False
        


        log_count += 1

        # compute sequences for classification report
        for batch_idx in range(len(samples)):
            seq_true = generate_sequence_targets(targets[batch_idx], timestamps)
            d = {
                'pred_boxes': outputs['pred_boxes'][batch_idx].detach().cpu(),
                'pred_logits': outputs['pred_logits'][batch_idx].detach().cpu()
            }
            seq_hat = generate_sequence_predictions(d, timestamps)

            y_true.append(seq_true)
            y_hat.append(seq_hat)

    #print(f"batch size {batch_size}") 
    #print(f"avg over batches: {acc_time / (num_batches * batch_size)}")
        

    # gather the stats from all processes
    #print("Averaged stats:", metric_logger)
    logging.info(f"Averaged stats:")
    logging.info(metric_logger)

    # log and print classification report
    y_true = np.array(y_true).flatten()
    y_hat = np.array(y_hat).flatten()

    #target_names = ['Fixation', 'Saccade', 'Blink']

    #c_report = classification_report(y_true, y_hat, target_names=target_names, output_dict=True)
    #for label in ['Fixation', 'Saccade', 'Blink', 'macro avg', 'weighted avg']:
    #   log = {f'{label}_{k}_valid':c_report[label][k] for k in c_report[label]}
    #   wandb.log(log)

    #wandb.log({'overall_accuracy_valid': c_report['accuracy']})

    #print('VALID CLASSIFICATION REPORT:\n', classification_report(y_true, y_hat, digits=4))
    logging.info(f"VALID CLASSIFICATION REPORT: \n {classification_report(y_true, y_hat, digits=4)}")

    print('Confusion Matrix Valid:')
    cm = confusion_matrix(y_true, y_hat)
    # cm = cm/cm.astype(np.float).sum(axis=1)
    logging.info(f"Confusion matrix: \n{cm}")

    #print(tabulate(cm, headers=['fix', 'sacc', 'blink']))
    #print()

    # TODO: return important stats
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
