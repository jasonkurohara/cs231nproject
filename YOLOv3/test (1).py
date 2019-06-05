from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    print(path)
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    print(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
  #  M1102, M1201, M1202, -1302
  #  f = open("M01102.txt","w+")
    x = path.split("/")
  #  x1 = x[2].split(".")
    f = open(x[2][1:],"w+")
    k = 0
    counter = 0
    run_times = []
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        
        # Extract labels
        for i in range(len(targets[:,1])):
            if (int(targets[i,1])==5) or (int(targets[i,1])==6) or (int(targets[i,1])==7):
                targets[i,1] = 2.000
                
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size
        #print(targets)
        

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            start = time.time()
            outputs = model(imgs)
            #print(outputs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            #print("after nonmax")
            end = time.time()
            print("TIME FOR ONE BATCH")
            print(str(end-start))
            counter += 1
            run_times.append(end-start)
           # print(outputs)
    print(run_times)  
    
    tot = 0
    for i in range(len(run_times)):
        rate = 8.0/float(run_times[i])
        tot+= rate
    print("AVERAGE RUN TIME FOR SCENE")
    print(str(tot/len(run_times)))
        
            #print(outputs)
        #print(targets)
      #  print(outputs)
#         sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
#         print(type(outputs))
        
        
        
#         for i in range(len(outputs)):
#             k += 1
#             out_img = outputs[i]
            
#             out_img = rescale_boxes(out_img, img_size, [540,1024])
#             for j in range(len(out_img)):
#                 output = out_img[j]
#                 left = output[0].item()
#                 top = output[1].item()
#                 wid = output[2].item()-output[0].item()
#                 hi = output[3].item()-output[1].item()
#                 conf = output[4].item()
#                 f.write("%d,-1,%.5f,%.5f,%.5f,%.5f,%.5f,-1,-1,-1\n" % (k, left, top, wid, hi, conf))
        
#     f.close()

    #ORIGINAL
  #  Concatenate sample statistics
#     true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
#     precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

   # print(sample_metrics)
#     true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
#     precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    return
    #return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

  #  print("Compute mAP...")
    
#     valid_path = []
#     for i in range(40):
#         valid_path.append(data_config["valid"+str(i+1)])
#        # print(
#        #if i<= 3:
#         #    continue
#         if i<= 7:
#             continue
#         precision, recall, AP, f1, ap_class = evaluate(
#             model,
#             path=valid_path[i],
#             iou_thres=opt.iou_thres,
#             conf_thres=opt.conf_thres,
#             nms_thres=opt.nms_thres,
#             img_size=opt.img_size,
#             batch_size=opt.batch_size,
#         )

  #  start = time.time()
    evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
    )
#  #   end = time.time()
  #  print("AVErAGE TIME: " +  str(1550.0/(end-start)))
    
    #############ORIGINAL
#     print("Average Precisions:")
#     for i, c in enumerate(ap_class):
#         print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]} - RC: {recall[i]} - P: {precision[i]} - f1: {f1[i]}")
        
    
    
#     for i, c in enumerate(ap_class):
#         print(f"+ Class '{0}' ({class_names}) - AP: {AP} - RC: {recall} - P: {precision} - f1: {f1}")

   # print(f"mAP: {AP.mean()}")
#
