from tqdm import tqdm
import torch
import torch.nn.functional as F

class ContTrainer:
    def __init__(self, dataloader, model, optimizer, recorder, gpu_ids,opts=None):
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.recorder = recorder
        if gpu_ids is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:%d' % gpu_ids[0])
        self.to_cuda = ['keypoints', 'insole','contact_label','contact_smpl']

        self.init_lr = opts.lr
        self.num_train_epochs = opts.num_train_epochs
        self.epochs = opts.epochs

        self.w_press = opts.w_press
        self.w_cont = opts.w_cont

        # Early stopping (opt-in via config; 0 disables). The loss curve of
        # this trainer flattens long before the LR schedule ends, so the
        # trailing epochs are ~zero-learning-rate iterations.
        self.estop_patience = getattr(opts, 'early_stop_patience', 0)
        self.estop_min_delta = getattr(opts, 'early_stop_min_delta', 0.0)
        self.estop_lr_floor = getattr(opts, 'early_stop_lr_floor', 0.0)

    def adjust_learning_rate(self, optimizer, epoch):
        """
        Sets the learning rate to the initial LR decayed by x every y epochs
        x = 0.1, y = args.num_train_epochs = 100
        """
        lr = self.init_lr * (0.1 ** (epoch // self.num_train_epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        self.recorder.init()
        best_loss = None
        patience_left = self.estop_patience
        for epoch in range(self.epochs):
            iter = 0
            loss_ls,loss_press_ls,loss_cont_ls = [],[],[]
            self.adjust_learning_rate(self.optimizer, epoch)
            for data in tqdm(self.dataloader):
                log = {}
                for data_item in self.to_cuda:
                    data[data_item] = data[data_item].to(device=self.device)
                pred_press, pred_cont = self.model(keypoints=data['keypoints'])
                pred_press = pred_press.squeeze(-1)
                pred_cont = pred_cont.squeeze(-1)
                data.update({'pred_press': pred_press,'pred_cont':pred_cont})

                loss_press = self.w_press*F.mse_loss(pred_press, data['insole'])
                # pred_cont is 2*96 SMPL-foot contact; contact_label is the
                # 31x22 insole mask (484 valid pixels).  The upstream target
                # caused a 192-vs-484 BCE shape mismatch.
                loss_cont = self.w_cont * F.binary_cross_entropy(
                    pred_cont, data['contact_smpl'])

                loss = loss_press + loss_cont

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                log.update({
                    'data': data,
                    'net' : self.model,
                    'optim' : self.optimizer,
                    'loss_press' : loss_press,
                    'loss_cont':loss_cont,
                    'loss':loss,
                    'epoch' : epoch,
                    'iter':iter,
                    'img_path':data['case_name']
                })
                loss_ls.append(loss.item())
                loss_press_ls.append(loss_press.item())
                loss_cont_ls.append(loss_cont.item())
                self.recorder.logPressNetTensorBoard(log)
                iter+=1
            loss_mean = torch.mean(torch.tensor(loss_ls))

            print('Epoch[%d]: total loss[%f],pressure loss[%f], contact loss[%f]'%(
                epoch,loss_mean,torch.mean(torch.tensor(loss_press_ls)),torch.mean(torch.tensor(loss_cont_ls))) )
            self.recorder.log(log)

            # ---- early stopping (only when enabled in the config) ----
            if self.estop_patience > 0:
                if best_loss is None or loss_mean < best_loss - self.estop_min_delta:
                    best_loss = loss_mean
                    patience_left = self.estop_patience
                    torch.save(self.model.state_dict(),
                               '%s/%s/net_epoch_best' % (
                                   self.recorder.checkpoint_path,
                                   self.recorder.name))
                else:
                    patience_left -= 1
                if patience_left <= 0:
                    print('Early stop at epoch %d: no loss improvement for '
                          '%d epochs (best %f)' % (
                              epoch, self.estop_patience, best_loss))
                    break
                lr = self.init_lr * (0.1 ** (epoch // self.num_train_epochs))
                if self.estop_lr_floor > 0 and lr < self.estop_lr_floor \
                        and epoch >= self.num_train_epochs:
                    print('Early stop at epoch %d: lr %g below floor %g' % (
                        epoch, lr, self.estop_lr_floor))
                    break
