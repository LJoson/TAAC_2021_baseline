from __future__ import print_function

from mmcv import Config
from tensorboardX import SummaryWriter

import src.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src import get_data
from torch.utils.data import DataLoader
from utilis import (cal_MIOU, cal_Recall, cal_Recall_time, get_ap, get_mAP_seq,
                    load_checkpoint, mkdir_ifmiss, pred2scene, save_checkpoint,
                    save_pred_seq, scene2video, to_numpy, write_json)
from utilis.package import *
import glob


final_dict = {}
test_iter, val_iter = 0, 0

def test(cfg, model, test_loader, criterion, mode='test'):
    global test_iter, val_iter
    model.eval()
    test_loss = 0
    correct1, correct0 = 0, 0
    gt1, gt0, all_gt = 0, 0, 0
    prob_raw, gts_raw = [], []
    preds, gts = [], []
    batch_num = 1	#0
    with torch.no_grad():
        for data_place, data_cast, data_act, data_aud, target in test_loader:
            batch_num += 1
            data_place = data_place.cuda() if 'place' in cfg.dataset.mode or 'image' in cfg.dataset.mode else []
            data_cast  = data_cast.cuda()  if 'cast'  in cfg.dataset.mode else []
            data_act   = data_act.cuda()   if 'act'   in cfg.dataset.mode else []
            data_aud   = data_aud.cuda()   if 'aud'   in cfg.dataset.mode else []
            target = target.view(-1).cuda()
            output = model(data_place, data_cast, data_act, data_aud)
            output = output.view(-1, 2)
            loss = criterion(output, target)

            if mode == 'test':
                test_iter += 1
                if loss.item() > 0:
                    writer.add_scalar('test/loss', loss.item(), test_iter)
            elif mode == 'val':
                val_iter += 1
                if loss.item() > 0:
                    writer.add_scalar('val/loss', loss.item(), val_iter)

            test_loss += loss.item()
            output = F.softmax(output, dim=1)
            prob = output[:, 1]
            gts_raw.append(to_numpy(target))
            prob_raw.append(to_numpy(prob))

            gt = target.cpu().detach().numpy()
            prediction = np.nan_to_num(prob.squeeze().cpu().detach().numpy()) > 0.5
            idx1 = np.where(gt == 1)[0]
            idx0 = np.where(gt == 0)[0]
            gt1 += len(idx1)
            gt0 += len(idx0)
            all_gt += len(gt)
            correct1 += len(np.where(gt[idx1] == prediction[idx1])[0])
            correct0 += len(np.where(gt[idx0] == prediction[idx0])[0])
        for x in gts_raw:
            gts.extend(x.tolist())
        for x in prob_raw:
            preds.extend(x.tolist())

    test_loss /= batch_num
    ap = get_ap(gts_raw, prob_raw)
    mAP, mAP_list = get_mAP_seq(test_loader, gts_raw, prob_raw)
    print("AP: {:.3f}".format(ap))
    print('mAP: {:.3f}'.format(mAP))
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct1 + correct0, all_gt,
        100. * (correct0 + correct1)/all_gt))
    print('Accuracy1: {}/{} ({:.0f}%), Accuracy0: {}/{} ({:.0f}%)'.format(
        correct1, gt1, 100.*correct1/(gt1+1e-5), correct0, gt0,
        100.*correct0/(gt0+1e-5)))
    if mode == "val" or mode == "test":
        return mAP.mean()
    elif mode == "test_final":
        final_dict.update({
            "AP":  ap,
            "mAP": mAP,
            "Accuracy":  100 * (correct0 + correct1)/all_gt,
            "Accuracy1": 100 * correct1/(gt1+1e-5),
            "Accuracy0": 100 * correct0/(gt0+1e-5),
            })
        return gts, preds

def load_model(cfg):
    model = models.__dict__[cfg.model.name](cfg).cuda()
    model = nn.DataParallel(model)
    checkpoint = load_checkpoint(cfg.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model 
    
def main(cfg, model):
    _, testSet, _ = get_data(cfg)
    test_loader = DataLoader(
                    testSet, batch_size=cfg.batch_size,
                    shuffle=False, **cfg.data_loader_kwargs)
    criterion = nn.CrossEntropyLoss(torch.Tensor(cfg.loss.weight).cuda())
    gts, preds = test(cfg, model, test_loader, criterion, mode='test_final')
    save_pred_seq(cfg, test_loader, gts, preds)

    print('...visualize scene video in demo mode', 'the above quantitive metrics are invalid')
    scene_dict, scene_list = pred2scene(cfg, threshold=0.65)
    scene2video(cfg, scene_list)
        
def parse_args():
    parser = argparse.ArgumentParser(description='Runner')
    parser.add_argument('--config', help='config file path', default = 'config/common_test.py')
    parser.add_argument('--data_root', help='data root', default = '../data/ad300')
    parser.add_argument('--video_dir', help='test video dir', default = None)
    parser.add_argument('--model_path', help='model root', default = None)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus    
    assert cfg.testFlag, "testFlag must be True"
    cfg.data_root = args.data_root if args.data_root is not None else cfg.data_root
    cfg.video_dir = args.video_dir
    cfg.model_path = args.model_path if args.model_path is not None else cfg.model_path
    cfg.shot_frm_path = os.path.join(cfg.data_root, 'shot_txt')
    
    model = load_model(cfg)
    print('model complete')
    for video_path in glob.glob(os.path.join(args.video_dir,'*.mp4')):
        cfg.video_name = os.path.basename(video_path).split(".m")[0]
        cfg.logger.logs_dir = os.path.join(args.data_root,"test_results", cfg.video_name)
        main(cfg, model)
