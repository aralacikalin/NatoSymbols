import argparse
import glob
import inspect
import json
import os
import sys
import time
from typing import Optional

from pathlib import Path
import cv2


import torchvision
import torch
import numpy as np
from tqdm import tqdm

from utils.plots import Annotator, colors

from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size, check_requirements,
                           check_yaml, coco80_to_coco91_class, colorstr, increment_path, non_max_suppression,
                           print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou,ap_all,ap_per_class_with_confidence,p_r_at
from utils.plots import output_to_target, plot_images, plot_val_study


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def process_prediction_text(predictions_path:str):

    imageLabelsDict={}
    textPaths=glob.glob(predictions_path+"/*.txt")
    for path in textPaths:

        nameImage=os.path.basename(path).split(".")[0]
        labelsTemp=[]

        with open(path,"r") as labels:
            labelList=labels.readlines()
            for l in labelList:
                l=l.split(" ")
                l=[float(a) for a in l]
                l=[l[1],l[2],l[3],l[4],l[5],l[0]]
                labelsTemp.append(l)

        imageLabelsDict[nameImage]=torch.tensor(labelsTemp)
        
    return imageLabelsDict
        





def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def PlotImg(det,path,save_dir,names,correctness,conf_tresh=0.0001):
        p=path
        im0=cv2.imread(path)
        # if np.max(im0) <= 1:
        #     im0 *= 255  # de-normalise (optional)
        # im0=cv2.cvtColor(im0,cv2.COLOR_GRAY2RGB)
        save_path = str(save_dir / "detections" / p.name)  # im.jpg
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc =im0  # for save_crop
        annotator = Annotator(im0, line_width=3, example=str(names))
        if len(det):

            # Write results
            for i,(*xyxy, conf, cls) in enumerate(det.tolist()):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                if(conf_tresh>conf):
                    continue


                if True:  # Add bbox to image
                    c = int(cls)  # integer class
                    label =f'{names[c]} {conf:.2f}'
                    iouExceeded=correctness[i]

                    annotator.box_label_iou(xyxy, label, color=colors(c, True),bbox_iou_exceeded=iouExceeded)


        # Stream results
        im0 = annotator.result()
    
        # Save results (image with detections)
        cv2.imwrite(save_path, im0)



def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    LOGGER.info(colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,
        nc=31,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = 0  # batch size
    nc = nc  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
            # break  # time limit exceeded

    return output




def run(
        data,
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        predictions=""
):
    # Initialize/load model and set device
    device = torch.device("cpu")

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / 'detections' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / "modelInputOutput" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    batch_size = 1

    # Data
    data = check_dataset(data)  # check

    # Configure
    cuda = device.type != 'cpu'
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    # iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.8:0.95
    # iouv = torch.linspace(0.95, 0.95, 1, device=device)  # iou vector for mAP@0.95:0.95
    niou = iouv.numel()

    # Dataloader
    dataloader = create_dataloader(data[task],
                                       imgsz,
                                       1,
                                       1,
                                       single_cls,
                                       pad=0,
                                       rect=True,
                                       workers=2,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc, conf=conf_thres,iou_thres=0.5)
    names = data['names']
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    allTreshStats=[]
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    default_conf_thres=0.001
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        imagepath=paths[0]
        imageName=os.path.basename(imagepath).split(".")[0]
        imagePredictionDict=process_prediction_text(predictions)

        callbacks.run('on_val_batch_start')
        with dt[0]:
            im = im.half() if half else im.float()  # uint8 to fp16/32
            nb, _, height, width = im.shape  # batch size, channels, height, width


        try:
            preds=imagePredictionDict[imageName]
            preds[:,:-2] *= torch.tensor((width, height, width, height), device=device)  # to pixels

            preds[:,:-2]=xywh2xyxy(preds[:,:-2])
            # preds[:,:-2] *= torch.tensor((height,width,  height,width), device=device)  # to pixels
            preds=[preds]
        except:
            
            preds=[torch.tensor([])]

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        
        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]




            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
                PlotImg(predn,path,save_dir,names,correct[:,0],conf_thres)
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        allThresholdPreds=preds.copy()
        for si, pred in enumerate(allThresholdPreds):
            labels = targets[targets[:, 0] == si, 1:]

            #! for merging the screen cover guard
            """for y,l in enumerate(labels):
                if(l[0]==16 or l[0]==24): # label on the label
                    labels[y][0]=10
            for y,l in enumerate(pred):
                if(l[5]==16 or l[5]==24): #label prediction on the tensor
                    pred[y][5]=10"""


            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init

            if npr == 0:
                if nl:
                    allTreshStats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))

                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            allTreshStats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)


        # Plot images
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
            plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

        if(save_txt):
            for i, det in enumerate(preds):  # per image
                im0=im[i,0].cpu().float().numpy()
                if np.max(im0) <= 1:
                    im0 *= 255  # de-normalise (optional)
                im0=cv2.cvtColor(im0,cv2.COLOR_GRAY2RGB)
                p = Path(paths[i])  # to Path
                save_path = str(save_dir / "modelInputOutput" / p.name)  # im.jpg
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc =im0  # for save_crop
                annotator = Annotator(im0, example=str(names))
                if len(det):

                    # Write results
                    for *xyxy, conf, cls in det.tolist():
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh


                        if True:  # Add bbox to image
                            c = int(cls)  # integer class
                            label =f'{names[c]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))


                # Stream results
                im0 = annotator.result()
            
                # Save results (image with detections)
                cv2.imwrite(save_path, im0)
               




        callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)


    # Compute metrics
    statsTemp=stats.copy()
    stats=allTreshStats
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        isConfidenceReturned=True
        try:
            tp, fp, p, r, f1, ap, ap_class,f1MaxConfThreshold = ap_per_class_with_confidence(*stats, plot=plots, save_dir=save_dir, names=names)
        except:
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
            fpC=fp.sum()
            isConfidenceReturned=False

        tpMicro, fpMicro, pMicro, rMicro, f1Micro, apMicro, _,f1MaxConfThresholdMicro = ap_all(*stats, plot=plots, save_dir=save_dir, names=names)
        pAtTarget,rAtTarget, confRatP, confPatR= p_r_at(*stats, plot=plots, save_dir=save_dir, names=names)

        print(f"P@R(80) Conf: {confPatR}, R@P(80) Conf: {confRatP}")

        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        map50Micro, mapMicro = apMicro[:, 0], apMicro.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)
    tpC=tp.sum()
    tpMicroC=tpMicro.sum()
    fpMicroC=fpMicro.sum()
    if(isConfidenceReturned):
        fpC=0  # number of targets per class
        predictionCorrectness,conf,predClass,_=stats
        indexesConfidenceBigger=np.argwhere(conf>=f1MaxConfThresholdMicro)
        predictionCorrectness=predictionCorrectness[indexesConfidenceBigger]
        fpC=len(predictionCorrectness)-tpMicroC
        LOGGER.info(f'Results at max f1_score, confidence threshold: {f1MaxConfThreshold}')
    else:
        LOGGER.info(f'Results at max f1_score')


    LOGGER.info(('%22s' + '%11s' * 9) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP80', 'mAP80-95',"P@R({80})","R@P({80})","MaxF1"))
    resultsToSave = f"P@R(80) Conf: {confPatR}, R@P(80) Conf: {confRatP}\n"
    resultsToSave += ('%22s' + '%11s' * 9) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP80', 'mAP80-95',"P@R({80})","R@P({80})","MaxF1") +"\n"
    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    pf2 = '%22s' + '%11i' * 2 + '%11.3g' * 7  # print format
    LOGGER.info(pf % ('macro average', seen, nt.sum(), mp, mr, map50, map))
    resultsToSave+=pf % (f'macro average', seen, nt.sum(), mp, mr, map50, map) +  f" @{f1MaxConfThreshold}" +"\n"
    resultsToSave+=pf2 % (f'micro average', seen, nt.sum(), (pMicro.sum()), (rMicro.sum()), map50Micro, mapMicro,pAtTarget,rAtTarget,f1Micro[0])+  f" @{f1MaxConfThresholdMicro}" +"\n"
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')
    LOGGER.info(pf2 % (f'micro average', seen, nt.sum(), (pMicro.sum()), (rMicro.sum()), map50Micro, mapMicro,pAtTarget,rAtTarget,f1Micro[0]))
    # Print results per class
    if (verbose or (nc < 50)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
            resultsToSave+=pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]) +"\n"


    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    shape = (batch_size, 3, imgsz, imgsz)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)
    resultsToSave+=f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t
    
    if(isConfidenceReturned):

        resultsToSave+=f"\nConfidence Threshold = {f1MaxConfThreshold}, IoU Threhsold= {iou_thres}"
    else:
        resultsToSave+=f"\nIoU Threhsold= {iou_thres}"


    with open(f'{save_dir}/MaxF1PrecisionRecall.txt',"w") as resultsText:
        resultsText.write(resultsToSave)

    LOGGER.info(f'Results at {conf_thres}')

    stats=statsTemp
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    tp,fp=confusion_matrix.tp_fp()
    tpC=tp.sum()
    fpC=fp.sum()
    p=tp / (tp + fp+1e-16)
    r = tp / (nt+1e-16)
      # recall curve


    mp,mr=p[ap_class].mean(), r[ap_class].mean()
    resultsToSave = ('%22s' + '%11s' * 4) % ('Class', 'Images', 'Instances', 'P', 'R') +"\n"
    LOGGER.info(('%22s' + '%11s' * 4) % ('Class', 'Images', 'Instances', 'P', 'R'))
    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 2  # print format
    LOGGER.info(pf % ('macro average', seen, nt.sum(), mp, mr))
    resultsToSave+=pf % ('macro average', seen, nt.sum(), mp, mr) +"\n"
    resultsToSave+=pf % ('micro average', seen, nt.sum(), (tpC/(tpC+fpC)), (tpC/(nt.sum()+1e-16))) +"\n"
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')
    LOGGER.info(pf % ('micro average', seen, nt.sum(), (tpC/(tpC+fpC)), (tpC/(nt.sum()+1e-16))))
    # Print results per class
    if (verbose or (nc < 50)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[c], r[c]))
            resultsToSave+=pf % (names[c], seen, nt[c], p[c], r[c]) +"\n"

    resultsToSave+=f"\nConfidence Threshold = {conf_thres}, IoU Threhsold= {iou_thres}"
    with open(f'{save_dir}/PrecisionRecall.txt',"w") as resultsText:
        resultsText.write(resultsToSave)

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    shape = (batch_size, 3, imgsz, imgsz)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)
    resultsToSave+=f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t
    

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()),normalize=False)
        callbacks.run('on_val_end', nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

   

    # Return results
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--predictions', type=str,default=ROOT / 'labels', help='predictions of the model in text file with confidences.')

    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')
        
        # process_prediction_text(opt.predictions)

        # exit()

        run(**vars(opt))

   



if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

# example usage;
# python .\standAloneVal.py --predictions D:\Workplace\Symbols\YOLO-Detection\yolov5\runs\1280\FinalDataModels\Yolov8-labels --data "D:\Workplace\Symbols\YOLO-Detection\yolov5-old\data\NATO-Symbols-31TestRealData.yaml" --save-txt --project "D:\Workplace\Symbols\YOLO-Detection\yolov5\runs\1280\TESTSTANDALONE" --conf-thres 0.7