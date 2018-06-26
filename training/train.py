import os
import sys
sys.path.append('../lib')
import torch.backends.cudnn as cudnn
import torch.optim
import argparse
import time
from nets.network import *
from loss.loss_function import L2_loss
from utils.Mydataloader import CPNFolder
import utils.Mytransforms as Mytransforms



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def parse():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', default='/root/Desktop/data',
                        nargs='+', type=str, dest='data_root', help='the path of train file')
    parser.add_argument('--train_txt', default='/txt/merged_txt/20180607_train_930k_3_1_10.txt',
                        nargs='+', type=str, dest='train_txt', help='the path of train file')
    parser.add_argument('--val_txt', default='/txt/merged_txt/20180607_valid_1k.txt',
                        nargs='+', type=str, dest='val_txt', help='the path of val file')

    parser.add_argument('--gpu', default=[0], nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--pretrained', default=None,type=str,
                        dest='pretrained', help='the path of pretrained model')
    parser.add_argument('--output_shape', type=list, default=[80, 80],
                        dest='output_shape', help='')
    parser.add_argument('--input_shape', type=list, default=[320, 320],
                        dest='input_shape', help='')
    parser.add_argument('--batch_size', type=int, default=30,
                        dest='batch_size', help='')
    parser.add_argument('--num_points', type=int, default=24,
                        dest='num_points', help='')
    parser.add_argument('--top_k', type=int, default=18,
                        dest='top_k', help='')
    parser.add_argument('--base_lr', type=float, default=5e-4,
                        dest='base_lr', help='')
    parser.add_argument('--max_iter', type=int, default=1800000,
                        dest='max_iter', help='')
    parser.add_argument('--display', type=int, default=1000,
                        dest='display', help='')
    parser.add_argument('--test_interval', type=int, default=10000,
                        dest='test_interval', help='')
    parser.add_argument('--workers', type=int, default=2,
                        dest='workers', help='')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        dest='weight_decay', help='')
    parser.add_argument('--momentum', type=float, default=0.5,
                        dest='momentum', help='')
    parser.add_argument('--start_iters', type=int, default=0,
                        dest='start_iters', help='')

    return parser.parse_args()

def construct_model(args, device):

    if args.pretrained is not None:
        model = CPN(args.output_shape, args.num_points, pretrained=False)
        
        state_dict = torch.load(args.pretrained)['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        state_dict = model.state_dict()
        state_dict.update(new_state_dict)
        model.load_state_dict(state_dict)
    else:
        model = CPN(args.output_shape, args.num_points)
    model = torch.nn.DataParallel(model, device_ids=args.gpu)

    return model.to(device)

def get_parameters(model, args, isdefault=True):

    if isdefault:
        return model.parameters(), [1.]
    lr_1 = []
    lr_2 = []
    params_dict = dict(model.module.named_parameters())
    for key, value in params_dict.items():
        if key[-4:] == 'bias':
            lr_2.append(value)
        else:
            lr_1.append(value)
    params = [{'params': lr_1, 'lr': args.base_lr},
            {'params': lr_2, 'lr': args.base_lr * 2.}]

    return params, [1., 2.]

def train_val(model,device, args):
    data_root = args.data_root
    train_txt = data_root+args.train_txt
    val_txt = data_root+args.val_txt

    cudnn.benchmark = True
    
    train_loader = torch.utils.data.DataLoader(
            CPNFolder(train_txt, args.output_shape,
                Mytransforms.Compose([Mytransforms.RandomResized(),
                Mytransforms.RandomRotate(40),
                Mytransforms.RandomCrop(320),
                Mytransforms.RandomHorizontalFlip(),
            ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

    if args.test_interval != 0 and args.val_txt is not None:
        val_loader = torch.utils.data.DataLoader(
                CPNFolder(val_txt, args.output_shape,
                    Mytransforms.Compose([Mytransforms.TestResized(320),
                ])),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
    
    params, multiple = get_parameters(model, args, False)
    
    optimizer = torch.optim.SGD(params, args.base_lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    global_losses = [AverageMeter() for i in range(4)]
    refine_losses = AverageMeter()
    
    end = time.time()
    iters = args.start_iters
    learning_rate = args.base_lr

    model.train()

    while iters < args.max_iter:
    
        for i, (input, label15, label11, label9, label7, valid) in enumerate(train_loader):

            #learning_rate = adjust_learning_rate(optimizer, iters, args.base_lr, policy=args.lr_policy, policy_parameter=args.policy_parameter, multiple=multiple)
            data_time.update(time.time() - end)

            label15 = label15.to(device)
            label11 = label11.to(device)
            label9 = label9.to(device)
            label7 = label7.to(device)
            valid = valid.to(device)

            labels = [label15, label11, label9, label7]

            global_out, refine_out = model(input)

            global_loss, refine_loss = L2_loss(global_out, refine_out, labels, valid, args.top_k, args.batch_size, args.num_points)
            
            loss = 0.0

            for i, global_loss1 in  (global_loss):
                loss += global_loss1
                global_losses[i].update(global_loss1.data[0], input.size(0))

            loss += refine_loss
            losses.update(loss.data[0], input.size(0))
            refine_losses.update(refine_loss.data[0], input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            batch_time.update(time.time() - end)
            end = time.time()
    
            iters += 1
            if iters % args.display == 0:
                print('Train Iteration: {0}\t'
                    'Time {batch_time.sum:.3f}s / {1}iters, ({batch_time.avg:.3f})\t'
                    'Data load {data_time.sum:.3f}s / {1}iters, ({data_time.avg:3f})\n'
                    'Learning rate = {2}\n'
                    'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                    iters, args.display, learning_rate, batch_time=batch_time,
                    data_time=data_time, loss=losses))
                for cnt in range(0,4):
                    print('Global Net Loss{0} = {loss1.val:.8f} (ave = {loss1.avg:.8f})'.format(cnt + 1, loss1=global_losses[cnt]))
                print('Refine Net Loss = {loss1.val:.8f} (ave = {loss1.avg:.8f})'.format(loss1=refine_losses))

                print(time.strftime('%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n', time.localtime()))

                batch_time.reset()
                data_time.reset()
                losses.reset()
                for cnt in range(4):
                    global_losses[cnt].reset()
                refine_losses.reset()
    
            if args.test_interval != 0 and args.val_dir is not None and iters % args.test_interval == 0:

                model.eval()
                for j, (input, label15, label11, label9, label7, valid) in enumerate(val_loader):
                    
                    label15 = label15.to(device)
                    label11 = label11.to(device)
                    label9 = label9.to(device)
                    label7 = label7.to(device)
                    valid = valid.to(device)
        
                    labels = [label15, label11, label9, label7]
        
                    global_out, refine_out = model(input)
        
                    global_loss, refine_loss = L2_loss(global_out, refine_out, labels,valid, args.top_k, args.batch_size, args.num_points)
                    
                    loss = 0.0
        
                    for i, global_loss1 in enumerate(global_loss):
                        loss += global_loss1
                        global_losses[i].update(global_loss1.data[0], input.size(0))
        
                    loss += refine_loss
                    losses.update(loss.data[0], input.size(0))
                    refine_losses.update(refine_loss.data[0], input.size(0))

                batch_time.update(time.time() - end)
                end = time.time()
                # save_checkpoint({
                #     'iter': iters,
                #     'state_dict': model.state_dict(),
                #     }, 'cpn_fashion')
    
                print(
                    'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
                    'Loss {loss.avg:.8f}\n'.format(
                    batch_time=batch_time, loss=losses))
                for cnt in range(0,4):
                    print('Global Net Loss{0} = {loss1.val:.8f} (ave = {loss1.avg:.8f})'.format(cnt + 1, loss1=global_losses[cnt]))
                print('Refine Net Loss = {loss1.val:.8f} (ave = {loss1.avg:.8f})'.format(loss1=refine_losses))
                print(time.strftime('%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n', time.localtime()))
    
                batch_time.reset()
                losses.reset()
                for cnt in range(4):
                    global_losses[cnt].reset()
                refine_losses.reset()
                
                model.train()
    
            if iters == args.max_iter:
                break


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    args = parse()
    device = torch.device("cuda")
    model = construct_model(args, device)
    train_val(model, device, args)
