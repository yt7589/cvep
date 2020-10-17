#
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
#
from apps.dcl.nnm.angle_loss import AngleLoss
from apps.dcl.nnm.focal_loss import FocalLoss
from apps.dcl.nnm.loss_record import LossRecord
from apps.dcl.nnm.dcl_util import DclUtil
from apps.dcl.controller.c_wxs import CWxs

class DclEngine(object):
    def __init__(self):
        self.refl = ''

    def train(self, config,
            model,
            epoch_num,
            start_epoch,
            optimizer,
            exp_lr_scheduler,
            data_loader,
            save_dir,
            data_size=448,
            savepoint=500,
            checkpoint=1000
            ):
        # savepoint: save without evalution
        # checkpoint: save with evaluation
        print('DclEngine.train 1')
        bmy_weight = 1.0 # 1.5 # 决定品牌分支在学习中权重
        step = 0
        eval_train_flag = False
        rec_loss = []
        checkpoint_list = []

        steps = np.array([], dtype=np.int)
        train_accs = np.array([], dtype=np.float32)
        test_accs = np.array([], dtype=np.float32)
        ce_losses = np.array([], dtype=np.float32)
        ce_loss_mu = -1
        ce_loss_std = 0.0

        train_batch_size = data_loader['train'].batch_size
        train_epoch_step = data_loader['train'].__len__()
        train_loss_recorder = LossRecord(train_batch_size)

        if savepoint > train_epoch_step:
            savepoint = 1*train_epoch_step
            checkpoint = savepoint

        date_suffix = DclUtil.datetime_format()
        log_file = open(os.path.join(config.log_folder, 'formal_log_r50_dcl_%s_%s.log'%(str(data_size), date_suffix)), 'a')

        add_loss = nn.L1Loss()
        get_ce_loss = nn.CrossEntropyLoss()
        get_focal_loss = FocalLoss()
        get_angle_loss = AngleLoss()

        print('DclEngine.train 2')

        inputs, brand_labels, img_names, bmy_labels, \
            brand_labels_swap, swap_law = None, None, \
            None, None, None, None
        org_brand_labels, swap_loss, law_loss = None, None, None
        print('DclEngine.train 3')
        for epoch in range(start_epoch,epoch_num-1):
            print('DclEngine.train 4: epoch:{0};'.format(epoch))
            model.train(True)
            save_grad = []
            for batch_cnt, data in enumerate(data_loader['train']):
                print('DclEngine.train 5: batch_cnt={0};'.format(batch_cnt))
                step += 1
                loss = 0
                model.train(True)
                if config.use_backbone:
                    inputs, brand_labels, img_names, bmy_labels = data
                    inputs = Variable(inputs.cuda())
                    brand_labels = Variable(torch.from_numpy(np.array(brand_labels)).cuda())
                    bmy_labels = Variable(torch.from_numpy(np.array(bmy_labels)).cuda())

                if config.use_dcl:
                    inputs, brand_labels, brand_labels_swap, swap_law, img_names, bmy_labels = data
                    org_brand_labels = brand_labels
                    inputs = Variable(inputs.cuda())
                    brand_labels = Variable(torch.from_numpy(np.array(brand_labels)).cuda())
                    bmy_labels = Variable(torch.from_numpy(np.array(bmy_labels)).cuda())
                    brand_labels_swap = Variable(torch.from_numpy(np.array(brand_labels_swap)).cuda())
                    swap_law = Variable(torch.from_numpy(np.array(swap_law)).float().cuda())
                print('DclEngine.train 6')
                optimizer.zero_grad()

                if inputs.size(0) < 2*train_batch_size:
                    outputs = model(inputs, inputs[0:-1:2])
                else:
                    outputs = model(inputs, None)
                print('DclEngine.train 7')

                if config.use_focal_loss:
                    ce_loss_brand = get_focal_loss(outputs[0], brand_labels)
                    ce_loss_bmy = get_focal_loss(outputs[-1], bmy_labels)
                else:
                    ce_loss_brand = get_ce_loss(outputs[0], brand_labels)
                    ce_loss_bmy = get_ce_loss(outputs[-1], bmy_labels)
                ce_loss = ce_loss_brand + bmy_weight * ce_loss_bmy
                print('DclEngine.train 8')

                if config.use_Asoftmax:
                    fetch_batch = brand_labels.size(0)
                    if batch_cnt % (train_epoch_step // 5) == 0:
                        angle_loss = get_angle_loss(outputs[3], brand_labels[0:fetch_batch:2], decay=0.9)
                    else:
                        angle_loss = get_angle_loss(outputs[3], brand_labels[0:fetch_batch:2])
                    loss += angle_loss
                print('DclEngine.train 9')

                loss += ce_loss
                ce_loss_val = ce_loss.detach().item()
                ce_losses = np.append(ce_losses, ce_loss_val)
                print('DclEngine.train 10')

                alpha_ = 1
                beta_ = 1
                gamma_ = 0.01 if config.dataset == 'STCAR' or config.dataset == 'AIR' else 1
                if config.use_dcl:
                    swap_loss = get_ce_loss(outputs[1], brand_labels_swap) * beta_
                    loss += swap_loss
                    law_loss = add_loss(outputs[2], swap_law) * gamma_
                    loss += law_loss
                print('DclEngine.train 11')

                loss.backward()
                torch.cuda.synchronize()
                optimizer.step()
                exp_lr_scheduler.step(epoch)
                torch.cuda.synchronize()
                print('DclEngine.train 12')


                if config.use_dcl:
                    print('DclEngine.train 13')
                    if ce_loss_mu > 0 and ce_loss_val > ce_loss_mu + 3.0*ce_loss_std:
                        # 记录下这个批次，可能是该批次有标注错误情况
                        print('记录可疑批次信息: loss={0}; threshold={1};'.format(ce_loss_val, ce_loss_mu + 2.0*ce_loss_std))
                        with open('./logs/abnormal_samples_{0}_{1}_{2}.txt'.format(epoch, step, ce_loss_val), 'a+') as fd:
                            error_batch_len = len(img_names)
                            for i in range(error_batch_len):
                                fd.write('{0} <=> {1};\r\n'.format(org_brand_labels[i*2], img_names[i]))
                    print('epoch{}: step: {:-8d} / {:d} loss=ce_loss+'
                                'swap_loss+law_loss: {:6.4f} = {:6.4f} '
                                '+ {:6.4f} + {:6.4f} brand_loss: {:6.4f}'.format(
                                    epoch, step % train_epoch_step, 
                                    train_epoch_step, 
                                    loss.detach().item(), 
                                    ce_loss_val, 
                                    swap_loss.detach().item(), 
                                    law_loss.detach().item(),
                                    ce_loss_brand.detach().item()), flush=True
                                )
                    
                if config.use_backbone:
                    print('DclEngine.train 14')
                    print('epoch{}: step: {:-8d} / {:d} loss=ce_loss+'
                                'swap_loss+law_loss: {:6.4f} = {:6.4f} '.format(
                                    epoch, step % train_epoch_step, 
                                    train_epoch_step, 
                                    loss.detach().item(), 
                                    ce_loss.detach().item()), flush=True
                                )
                rec_loss.append(loss.detach().item())

                train_loss_recorder.update(loss.detach().item())
                print('DclEngine.train 16')

                # evaluation & save
                if step % checkpoint == 0:
                    print('DclEngine.train 17')
                    rec_loss = []
                    print(32*'-', flush=True)
                    print('step: {:d} / {:d} global_step: {:8.2f} train_epoch: {:04d} rec_train_loss: {:6.4f}'.format(step, train_epoch_step, 1.0*step/train_epoch_step, epoch, train_loss_recorder.get_val()), flush=True)
                    print('current lr:%s' % exp_lr_scheduler.get_lr(), flush=True)
                    '''
                    if eval_train_flag:
                        trainval_acc1, trainval_acc2, trainval_acc3 = eval_turn(Config, model, data_loader['trainval'], 'trainval', epoch, log_file)
                        if abs(trainval_acc1 - trainval_acc3) < 0.01:
                            eval_train_flag = False
                    '''
                    print('##### validate dataset #####')
                    trainval_acc1, trainval_acc2, trainval_acc3 = self.evaluate(
                        config, model, data_loader['val'], 'val', epoch, log_file
                    ) #eval_turn(Config, model, data_loader['trainval'], 'trainval', epoch, log_file)
                    print('##### test dataset #####')
                    val_acc1, val_acc2, val_acc3 = trainval_acc1, trainval_acc2, \
                                trainval_acc3 # eval_turn(Config, model, data_loader['val'], 'val', epoch, log_file)
                    steps = np.append(steps, step)
                    train_accs = np.append(train_accs, trainval_acc1)
                    test_accs = np.append(test_accs, val_acc1)

                    save_path = os.path.join(save_dir, 'weights_%d_%d_%.4f_%.4f.pth'%(epoch, batch_cnt, val_acc1, val_acc3))
                    torch.cuda.synchronize()
                    torch.save(model.state_dict(), save_path)
                    print('saved model to %s' % (save_path), flush=True)
                    torch.cuda.empty_cache()
                    # 保存精度等信息并初始化
                    ce_loss_mu = ce_losses.mean()
                    ce_loss_std = ce_losses.std()
                    print('Cross entropy loss: mu={0}; std={1}; range:{2}~{3};'.format(
                        ce_loss_mu, ce_loss_std, 
                        ce_loss_mu - 3.0*ce_loss_std,
                        ce_loss_mu + 3.0 * ce_loss_std
                    ))
                    ce_losses = np.array([], dtype=np.float32)
                    if train_accs.shape[0] > 30:
                        np.savetxt('./logs/steps1.txt', (steps,))
                        np.savetxt('./logs/train_accs1.txt', (train_accs,))
                        np.savetxt('./logs/test_accs1.txt', (test_accs,))
                        steps = np.array([], dtype=np.int)
                        train_accs = np.array([], dtype=np.float32)
                        test_accs = np.array([], dtype=np.float32)
                # save only
                elif step % savepoint == 0:
                    print('DclEngine.train 18')
                    train_loss_recorder.update(rec_loss)
                    rec_loss = []
                    save_path = os.path.join(save_dir, 'savepoint_weights-%d-%s.pth'%(step, dt()))

                    checkpoint_list.append(save_path)
                    if len(checkpoint_list) == 6:
                        os.remove(checkpoint_list[0])
                        del checkpoint_list[0]
                    torch.save(model.state_dict(), save_path)
                    torch.cuda.empty_cache()


        log_file.close()

    def log_progress(self, step, train_acc, test_acc):
        # 以添加形式保存step
        with open('./logs/step.txt', 'a+') as step_fd:
            step_fd.write('{0:d},'.format(step))
        # 以添加形式保存train_acc
        with open('./logs/train_acc.txt', 'a+') as train_acc_fd:
            train_acc_fd.write('{0:.4f},'.format(train_acc))
        # 以添加形式保存test_acc
        with open('./logs/test_acc.txt', 'a+') as test_acc_fd:
            test_acc_fd.write('{0:.4f},'.format(test_acc))

    def evaluate(config, model, data_loader, val_version, epoch_num, log_file, efd=None):
        model.train(False)
        val_corrects1 = 0
        val_corrects2 = 0
        val_corrects3 = 0
        bmy_correct = 0
        bm_correct = 0
        bb_correct = 0 # 通过bmy求出的品牌精度
        val_size = data_loader.__len__()
        item_count = data_loader.total_item_len
        t0 = time.time()
        get_l1_loss = nn.L1Loss()
        get_ce_loss = nn.CrossEntropyLoss()

        val_batch_size = data_loader.batch_size
        val_epoch_step = data_loader.__len__()
        num_cls = data_loader.num_cls

        val_loss_recorder = LossRecord(val_batch_size)
        val_celoss_recorder = LossRecord(val_batch_size)
        print('evaluating %s ...'%val_version, flush=True)
        #
        bmy_id_bm_vo_dict = CWxs.get_bmy_id_bm_vo_dict()
        #bmy_sim_org_dict = WxsDsm.get_bmy_sim_org_dict()
        with torch.no_grad():
            for batch_cnt_val, data_val in enumerate(data_loader):
                inputs = Variable(data_val[0].cuda())
                print('eval_model.eval_turn inputs: {0};'.format(inputs.shape))
                brand_labels = Variable(torch.from_numpy(np.array(data_val[1])).long().cuda())
                bmy_labels = Variable(torch.from_numpy(np.array(data_val[-1])).long().cuda())
                img_files  = data_val[-2]
                outputs = model(inputs)
                loss = 0

                ce_loss = get_ce_loss(outputs[0], brand_labels).item()
                loss += ce_loss

                val_loss_recorder.update(loss)
                val_celoss_recorder.update(ce_loss)

                if config.use_dcl and config.cls_2xmul:
                    outputs_pred = outputs[0] + outputs[1][:,0:num_cls] + outputs[1][:,num_cls:2*num_cls]
                else:
                    outputs_pred = outputs[0]
                top3_val, top3_pos = torch.topk(outputs_pred, 3)

                print('{:s} eval_batch: {:-6d} / {:d} loss: {:8.4f}'.format(val_version, batch_cnt_val, val_epoch_step, loss), flush=True)

                batch_corrects1 = torch.sum((top3_pos[:, 0] == brand_labels)).data.item()
                val_corrects1 += batch_corrects1
                batch_corrects2 = torch.sum((top3_pos[:, 1] == brand_labels)).data.item()
                val_corrects2 += (batch_corrects2 + batch_corrects1)
                batch_corrects3 = torch.sum((top3_pos[:, 2] == brand_labels)).data.item()
                val_corrects3 += (batch_corrects3 + batch_corrects2 + batch_corrects1)
                # 求出年款精度
                outputs_bmy = outputs[-1]
                bmy_top5_val, bmy_top5_pos = torch.topk(outputs_bmy, 5)
                batch_bmy_correct = torch.sum((bmy_top5_pos[:, 0] == bmy_labels)).data.item()
                bmy_correct += batch_bmy_correct
                bb_correct = 0
                # 求出车型精度
                batch_bm_correct = 0
                for im in range(bmy_top5_pos.shape[0]):
                    gt_bmy_id = bmy_top5_pos[im][0].item()
                    net_bmy_id = bmy_labels[im].item()
                    if gt_bmy_id in bmy_id_bm_vo_dict:
                        gt_bm_vo = bmy_id_bm_vo_dict[gt_bmy_id]
                        net_bm_vo = bmy_id_bm_vo_dict[net_bmy_id]
                        if gt_bm_vo['model_id'] == net_bm_vo['model_id']:
                            batch_bm_correct += 1
                bm_correct += batch_bm_correct
                # 找出品牌错误的样本，写入文件top1_error_samples
                if efd is not None:
                    for idx in range(top3_pos.shape[0]):
                        if top3_pos[idx][0] != brand_labels[idx]:
                            efd.write('{0}*{1}*{2}\n'.format(
                                img_files[idx], brand_labels[idx], 
                                top3_pos[idx][0]
                            ))
                '''
                # 
                pred_size = top3_pos[:, 0].shape[0]
                batch_bb_correct = 0
                for idx in range(pred_size):
                    pred_bmy = fgvc_id_brand_dict[int(top3_pos[idx][0])]
                    pred_brand = pred_bmy.split('_')[0]
                    gt_bmy = fgvc_id_brand_dict[int(labels[idx])]
                    gt_brand = gt_bmy.split('_')[0]
                    if pred_brand == gt_brand:
                        batch_bb_correct += 1
                bb_correct += batch_bb_correct
                brand_correct = 0
                '''

            val_acc1 = val_corrects1 / item_count
            val_acc2 = val_corrects2 / item_count
            val_acc3 = val_corrects3 / item_count
            bmy_acc = bmy_correct / item_count
            bm_acc = bm_correct / item_count
            bb_acc = bb_correct / item_count

            log_file.write(val_version  + '\t' +str(val_loss_recorder.get_val())+'\t' + str(val_celoss_recorder.get_val()) + '\t' + str(val_acc1) + '\t' + str(val_acc3) + '\n')


            t1 = time.time()
            since = t1-t0
            print('--'*30, flush=True)
            print('% 3d %s %s %s-loss: %.4f || 品牌：%s-acc@1: %.4f %s-acc@2: %.4f %s-acc@3: %.4f; 车型:%.4f; 年款：%.4f; ||time: %d' % (epoch_num, val_version, dt(), val_version, val_loss_recorder.get_val(init=True), val_version, val_acc1,val_version, val_acc2, val_version, val_acc3, bm_acc, bmy_acc, since), flush=True)
            print('--' * 30, flush=True)

        return val_acc1, val_acc2, val_acc3