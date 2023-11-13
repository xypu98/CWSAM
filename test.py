import argparse
import os

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

from torchvision import transforms
from mmcv.runner import load_checkpoint

import matplotlib.pyplot as plt
import numpy as np
import cv2

from PIL import Image

from eval_iou import SegmentationMetric

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable

def color_to_list(mask, palette=[ [1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,0,0]] ):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    #mask = mask.permute(1,2,0)
    mask = mask*255
    mask.int()
    semantic_map = np.zeros([1024,1024],dtype=np.int8)
    for i,colour in enumerate( palette):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map += class_map*int(i)


def onehot_to_mask(mask, palette=[ [1,0,0], [0,1,0], [0,0,1], [1,1,0],[0,0,0]]):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    mask = mask.permute(1,2,0).numpy()
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    #x=x.permute(2,0,1)
    #x=x.numpy()
    #x = np.around
    return x

def onehot_to_index_label(mask):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    mask = mask.permute(1,2,0).numpy()
    x = np.argmax(mask, axis=-1)
    #colour_codes = np.array(palette)
    #x = np.uint8(colour_codes[x.astype(np.uint8)])*255
    #x=x.permute(2,0,1)
    #x=x.numpy()
    #x = np.around
    return x


def de_normalize(image,mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]):
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    image=image*std+mean #由image=(x-mean)/std可知，x=image*std+mean
    image=image.numpy().transpose(1,2,0)  # 从tensor转为numpy,并由（C,H,W）变为（H,W,C）
    image=np.around(image * 255)  ##对数据恢复并进行取整
    image=np.array(image, dtype=np.uint8)  # 矩阵元素类型由浮点数转为整数
    return image


def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, config=None,config_name = None,
              verbose=False):
    model.eval()
    class_num = config['model']['args']['num_classes']
    color_palette = config['test_dataset']['dataset']['args']['palette']
    ignore_background =  config['test_dataset']['dataset']['args']['ignore_bg']
    #work_dir = config['work_dir'].split('/')[-1]
    work_dir = config_name
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
    elif eval_type == 'seg':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
        
        metric_seg = SegmentationMetric(class_num,  ignore_background)


    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')

    id = 0

    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']

        output_masks = model.infer(inp)
        #pred = torch.sigmoid(model.infer(inp))
        pred = torch.sigmoid(output_masks)

        for i in range(len(output_masks)):
            #print(len(batch_pred))
            output_masks[i]=output_masks[i].to('cpu') 
            pred[i]=pred[i].to('cpu')

        #output_masks = model.infer(inp)

        #pred = torch.sigmoid(output_masks)
        #output_masks = torch.where(pred>0.01, 1,pred)
        #output_masks = torch.where(output_masks <0.01,0,output_masks)
        #filter_output_masks=pred.permute(0,2,3,1).cpu().detach().numpy() > 0.88

        #output_masks =output_masks.permute(0,2,3,1).cpu().detach().numpy()*255
        #print(output_masks.size())
        #print(pred.size())
        #filter_output_masks = pred > 0.88
        #output_masks = pred[torch.as_tensor(filter_output_masks, device=pred.device)]
        #output_masks=output_masks.permute(0,2,3,1).cpu().detach().numpy()*255
        #output_masks = np.array(output_masks)

        output_mask = output_masks[0].cpu().detach()
        binary_mask = onehot_to_mask(output_mask,palette = color_palette)
        mask_index_label = onehot_to_index_label(output_mask).flatten()

        


        output_path = './save/'+work_dir+'/mask/'+str(id)+'.jpg'
        if not os.path.exists(output_path.split(str(id)+'.jpg')[0]):
            os.makedirs(output_path.split(str(id)+'.jpg')[0])
        #cv2.imwrite(output_path,output_masks[0])
        Image.fromarray(np.uint8(binary_mask)).convert('RGB').save(output_path)
        #plt.imsave(output_path,output_masks[0])


        #gt_mask = batch['gt'].permute(0,2,3,1).cpu().detach().numpy()*255
        gt_mask= batch['gt'][0].cpu().detach()
        gt_mask_rgb = onehot_to_mask(gt_mask,palette = color_palette)
        gt_index_label = onehot_to_index_label(gt_mask).flatten()


        #gt_mask=np.array(gt_mask)
        gt_save_path = './save/'+work_dir+'/gt/'+str(id)+'.jpg'
        if not os.path.exists(gt_save_path.split(str(id)+'.jpg')[0]):
            os.makedirs(gt_save_path.split(str(id)+'.jpg')[0])
        #cv2.imwrite(gt_save_path,gt_mask[0])
        Image.fromarray(np.uint8(gt_mask_rgb)).convert('RGB').save(gt_save_path)
        #plt.imsave(gt_save_path,gt_mask[0])

        #gt_img = batch['inp'].permute(0,2,3,1).cpu().detach().numpy()*255
        gt_img = batch['inp'][0].cpu().detach()
        #print(gt_img.shape)
        ori_gt_img = de_normalize(gt_img)
        #ori_gt_img = gt_img
        #gt_mask=np.array(gt_mask)
        img_save_path = './save/'+work_dir+'/gt_img/'+str(id)+'.jpg'
        if not os.path.exists(img_save_path.split(str(id)+'.jpg')[0]):
            os.makedirs(img_save_path.split(str(id)+'.jpg')[0])
        #cv2.imwrite(img_save_path,gt_img[0])
        #cv2.imwrite(img_save_path,ori_gt_img[0])
        Image.fromarray(np.uint8(ori_gt_img)).convert('RGB').save(img_save_path)

        overlay_mask_path = './save/'+work_dir+'/overlay_mask/'+str(id)+'.jpg'
        if not os.path.exists(overlay_mask_path.split(str(id)+'.jpg')[0]):
            os.makedirs(overlay_mask_path.split(str(id)+'.jpg')[0])
        overlay = Image.blend(Image.fromarray(np.uint8(ori_gt_img)).convert('RGB'), Image.fromarray(np.uint8(binary_mask)).convert('RGB'), alpha=0.5)
        overlay.save(overlay_mask_path)

        id+=1

        if eval_type == 'seg':
            metric_seg.addBatch(mask_index_label,gt_index_label)



        result1, result2, result3, result4 = metric_fn(pred, batch['gt'])

        val_metric1.add(result1.item(), inp.shape[0])
        val_metric2.add(result2.item(), inp.shape[0])
        val_metric3.add(result3.item(), inp.shape[0])
        val_metric4.add(result4.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))
            pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
            pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
            pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))

    oa = metric_seg.overallAccuracy()
    oa = np.around(oa,decimals=4)
    mIoU ,IoU= metric_seg.meanIntersectionOverUnion()
    mIoU = np.around(mIoU,decimals=4)
    IoU = np.around(IoU,decimals=4)
    p = metric_seg.precision()
    p = np.around(p,decimals=4)
    mp = np.nanmean(p)
    mp = np.around(mp,decimals=4)
    r = metric_seg.recall()
    r=np.around(r,decimals=4)
    mr = np.nanmean(r)
    mr = np.around(mr,decimals=4)
    f1 = (2*p*r) / (p + r)
    f1 = np.around(f1,decimals=4)
    mf1 = np.nanmean(f1)
    mf1 = np.around(mf1,decimals=4)
    normed_confusionMatrix = metric_seg.confusionMatrix / metric_seg.confusionMatrix.sum(axis=0)
    normed_confusionMatrix = np.around(normed_confusionMatrix, decimals=3)
    #print('total pixels:', metric.confusionMatrix.sum())
    #print('1024*1024*80=',1024*1024*80)
    fwIOU = metric_seg.Frequency_Weighted_Intersection_over_Union()
    fwIOU= np.around(fwIOU,decimals=4)
   
    classes_list = config['test_dataset']['dataset']['args']['classes']

    if ignore_background:
        axis_labels=classes_list[:-1] 
    else: 
        axis_labels=classes_list
    #axis_labels = ['building','vegetation','water','road'] #,'background']
    plt.figure()#figsize=(8, 8))
    sns.heatmap(normed_confusionMatrix, annot=True, cmap='Blues',yticklabels=axis_labels,xticklabels=axis_labels)

    #plt.ylim(0, 4)
    
    plt.ylabel('Predicted labels')
    plt.xlabel('True labels')
    #plt.yticks(np.array(range(0,5)), axis_labels)
    plt.savefig('./save/'+work_dir+'/confusionmatrix.jpg')
    #print('self.confusionMatrix:',metric.confusionMatrix / metric.confusionMatrix.sum(axis=0))


    #print(f' 类别0,类别1,...\n oa:{oa}, \n mIou:{mIoU}, \n p:{p}, \n mp:{mp},  \n r:{r}, \n mr:{mr}, \n f1:{f1}, \n mf1:{mf1}')
    print('self.confusionMatrix:')
    print(normed_confusionMatrix)
    print('OA:',oa)

    #print(IoU)
    #print(IoU.tolist())
    #print(mIoU)
    #print(['IOU',mIoU].extend(IoU.tolist()))
    IOU_row = ['IOU',mIoU]
    IOU_row.extend(IoU.tolist())
    Precision_row = ['Precision',mp]
    Precision_row.extend(p.tolist())
    Recall_row = ['Recall',mr]
    Recall_row.extend(r.tolist())
    F1_row = ['F1',mf1]
    F1_row.extend(f1.tolist())
    title_row = ['metrics','average']
    title_row.extend(axis_labels)
    OA_row = ['OA',oa]#,' ',' ',' ',' ']
    #OA_row.extend(' '*5)
    fwIOU_row = ['FWIOU', fwIOU]#,' ',' ',' ',' ']

    for i in range(len(axis_labels)):
        OA_row.append(' ')
        fwIOU_row.append(' ')

    table = PrettyTable(title_row)
    table.add_row(IOU_row)
    table.add_row(Precision_row)
    table.add_row(Recall_row)
    table.add_row(F1_row)
    table.add_row(OA_row)
    table.add_row(fwIOU_row)

    #print(table)

    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item(), table


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--prompt', default='none')
    args = parser.parse_args()

    config_name = args.config.split('/')[-1].split('.yaml')[0]
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8,shuffle= False)

    model = models.make(config['model']).cuda()
    sam_checkpoint = torch.load(args.model, map_location='cuda')
    model.load_state_dict(sam_checkpoint, strict=True)
    
    with torch.no_grad():
        metric1, metric2, metric3, metric4, seg_eval_table = eval_psnr(loader, model,
                                                   data_norm=config.get('data_norm'),
                                                   eval_type=config.get('eval_type'),
                                                   eval_bsize=config.get('eval_bsize'),
                                                   config=config,
                                                   config_name=config_name,
                                                   verbose=True)
    print('metric1: {:.4f}'.format(metric1))
    print('metric2: {:.4f}'.format(metric2))
    print('metric3: {:.4f}'.format(metric3))
    print('metric4: {:.4f}'.format(metric4))
    print(seg_eval_table)



