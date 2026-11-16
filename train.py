import time
import os
import argparse
from pathlib import Path
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random
from utils.utility import *
from utils.dataloader_newdata import *
from model.LF_DIDL import LF_DIDL
from utils.degrade1 import MultiDegrade
import utility
# from thop import profile


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--num_workers', type=int, default=4)  #参数指定了用于加载数据的子进程数量。
parser.add_argument('--model_name', type=str, default='LF-DIDL')
parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
parser.add_argument("--upfactor", type=int, default=4, help="upscale factor") # 超分辨率的放大因子
parser.add_argument('--load_pretrain', action='store_true', help='resume training from --model_path')
parser.add_argument('--model_path', type=str, default='weights/LF-DIDL_4xSR_epoch_100.tar')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--patchsize_train', type=int, default=32, help='patchsize of LR images for training')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')
parser.add_argument('--crop', type=bool, default=True)
parser.add_argument("--patchsize_test", type=int, default=32, help="patchsize of LR images for inference")
parser.add_argument("--minibatch_test", type=int, default=10, help="size of minibatch for inference")
parser.add_argument('--trainset_dir', type=str, required=True, help='directory containing training HDF5 files')


parser.add_argument('--testset_dir', type=str, required=True, help='directory of test dataset subfolders')

parser.add_argument('--epochs_encoder', type=int, default=50,
                    help='number of epochs to train the degradation encoder')

# Optimization specifications
parser.add_argument('--lr_encoder', type=float, default=1e-4,
                    help='learning rate to train the degradation encoder')
parser.add_argument('--lr_sr', type=float, default=2e-4,
                    help='learning rate to train the whole network')
parser.add_argument('--lr_decay_encoder', type=int, default=15,
                    help='learning rate decay per N epochs')
parser.add_argument('--lr_decay_sr', type=int, default=15,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma_encoder', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--gamma_sr', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--start_epoch', type=int, default=1200,
                    help='resume from the snapshot, and the start_epoch')


args = parser.parse_args()
os.chdir(Path(__file__).resolve().parent)


def train(args):
    net = LF_DIDL(args)
    net.to(args.device)

    cudnn.benchmark = True  #设置 cuDNN 基准测试为 True
    epoch_state = 0

    # # # 计算FLOPs和参数量
    # input = torch.randn(1, 5, 5, 3, 32, 32)
    # input1 = torch.randn(1, 5, 5, 3, 128, 128)
    # # net((data,label,False))
    # net.eval()
    # flops, params = profile(net,inputs=((input.to(args.device),input1.to(args.device),True),))
    # print(f'FLOPs: {flops / 1e9:.2f} GFLOPs, Params: {params / 1e6:.2f} M')

    if args.parallel:
        net = torch.nn.DataParallel(net, device_ids=[0, 1]) #这个对象会处理模型在多个GPU上的并行执行。

    criterion_Loss = torch.nn.L1Loss().to(args.device)
    # optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)
    # scheduler._step_count = epoch_state

    optimizer = utility.make_optimizer(args, net)
    scheduler = utility.make_scheduler(args, optimizer)
    if args.load_pretrain:
        model = torch.load(args.model_path, map_location=args.device)
        net.load_state_dict(model['state_dict'], strict=True)
        optimizer.load_state_dict(model['optimizer'])
        scheduler.load_state_dict(model['scheduler'])
        epoch_state = model['epoch']


    train_set = TrainSetLoader(args)
    
    # train_loader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)

    workers = min(args.num_workers, max(0, (os.cpu_count() or 1)-1))
    train_loader = DataLoader(
                                dataset=train_set,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=workers,
                                pin_memory=True,
                                persistent_workers=workers > 0,
                                **({'prefetch_factor': 4} if workers > 0 else {})
                            )
    
    # train_loader = DataLoaderX(dataset=train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=6)




    for idx_epoch in range(epoch_state, args.n_epochs):
        net.train()
        torch.cuda.empty_cache()

            # lr stepwise
        if (idx_epoch+1) <= args.epochs_encoder:
            lr = args.lr_encoder * (args.gamma_encoder ** (idx_epoch // args.lr_decay_encoder))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = args.lr_sr * (args.gamma_sr ** ((idx_epoch - args.epochs_encoder) // args.lr_decay_sr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


        # losses_diffuison, losses_sr  = utility.AverageMeter(), utility.AverageMeter()

        # 在GPU上统计指标（需自定义实现）
        losses_sr = 0
        losses_diffuison = 0
        count = 0


        
        
        for idx_iter, (data,label) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=70):
            
            label = label.to(args.device)
            data = data.to(args.device)


            # 切中间模块
            bdr = 12 // args.upfactor
            label1, data1 = label[:, :, :, :, 12:-12, 12:-12], data[:, :, :, :, bdr:-bdr, bdr:-bdr]

            ## 采用预先处理好的data和label来切块
            label, data = augmentation(label1, data1)
            

            optimizer.zero_grad()

            # forward
            ## train stage1
            if (idx_epoch+1) <= args.epochs_encoder:
                z, dwt2, sr = net((data,label,False))
                l_sr = criterion_Loss(sr, label)
                z = rearrange(z, '(b u v) c -> b (u v) c', u=args.angRes, v=args.angRes)
                l_cosine_similarity_loss =  cosine_similarity_loss(z)
                loss = l_sr + l_cosine_similarity_loss *0.01

                losses_sr += l_sr.detach()  # 不离开GPU
                
                # losses_sr.update(l_sr.item())
            ## train stage2
            else:
                z, dwt2, sr = net((data,label,True))
                l_diffusion =  criterion_Loss(z, dwt2)
                l_sr = criterion_Loss(sr, label)
                z = rearrange(z, '(b u v) c -> b (u v) c', u=args.angRes, v=args.angRes)
                l_cosine_similarity_loss =  cosine_similarity_loss(z)
                loss = l_diffusion  + l_sr *0.01 + l_cosine_similarity_loss *0.01

                losses_sr += l_sr.detach()
                losses_diffuison += l_diffusion.detach()

                # losses_diffuison.update(l_diffusion.item())
                # losses_sr.update(l_sr.item())

            count += 1

            # backward
            loss.backward()  # loss.backward()进行反向传播，计算梯度。
            optimizer.step() # optimizer.step()根据计算的梯度更新模型的权重。

        if (idx_epoch+1) <= args.epochs_encoder:
            print(time.ctime()[4:-5] + ' Epoch----%5d, loss_sr---%f' % (idx_epoch + 1, (losses_sr / count).item()))
            save_ckpt6(args, net, optimizer, scheduler, idx_epoch + 1)

            # if (idx_epoch+1) % 20 == 0:
            #     test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
            #     psnr_name = []
            #     ssim_name = []
            #     for index, test_name in enumerate(test_Names):
            #         torch.cuda.empty_cache()
            #         test_loader = test_Loaders[index]
            #         psnr_epoch_test, ssim_epoch_test = valid(test_loader, net, False)
            #         print('Dataset--%15s,\t PSNR--%f, \t SSIM---%f' % (
            #             test_name, psnr_epoch_test, ssim_epoch_test))
            #         psnr_name.append(psnr_epoch_test)
            #         ssim_name.append(ssim_epoch_test)

            #     psnr = float(np.array(psnr_name).mean())
            #     ssim = float(np.array(ssim_name).mean())
            #     print('PSNR--%.3f, \t SSIM---%.4f' % (psnr, ssim))

            #     # 写日志
            #     txtfile = open('./log/' + args.model_name + '_training_log.txt', 'a')
            #     txtfile.write('\n=== Epoch %d  PSNR: %.3f  SSIM: %.4f  \n\n' % (
            #         idx_epoch + 1, psnr, ssim))
            #     txtfile.close()
                # ----------------------------------------------------

        else:
            print(time.ctime()[4:-5] + ' Epoch----%5d, loss_sr---%f, losses_diffuison---%f' % (idx_epoch + 1, (losses_sr / count).item(), (losses_diffuison / count).item()))
            save_ckpt6(args, net, optimizer, scheduler, idx_epoch + 1)
            ''' evaluation '''
            if (idx_epoch+1) % 100 == 0:
                test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
                psnr_name = []
                ssim_name = []
                for index, test_name in enumerate(test_Names):
                    torch.cuda.empty_cache()
                    test_loader = test_Loaders[index]
                    psnr_epoch_test, ssim_epoch_test = valid(test_loader, net, True)
                    print('Dataset--%15s,\t PSNR--%f, \t SSIM---%f' % (
                        test_name, psnr_epoch_test, ssim_epoch_test))
                    psnr_name.append(psnr_epoch_test)
                    ssim_name.append(ssim_epoch_test)

                psnr = float(np.array(psnr_name).mean())
                ssim = float(np.array(ssim_name).mean())
                print('PSNR--%.3f, \t SSIM---%.4f' % (psnr, ssim))

                # 写日志
                os.makedirs('runs', exist_ok=True)
                txtfile = open('./runs/' + args.model_name + '_training_log.txt', 'a')
                txtfile.write('\n=== Epoch %d  PSNR: %.3f  SSIM: %.4f  \n\n' % (
                    idx_epoch + 1, psnr, ssim))
                txtfile.close()
                # ----------------------------------------------------
        
        scheduler.step()


def valid(test_loader, net, diffusion):
    net.eval()
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):

        if args.crop == False:
            with torch.no_grad():
                outLF = net((data.to(args.device), label.to(args.device), diffusion))
                outLF = outLF.squeeze()
        else:
            patch_size = args.patchsize_test
            data = data.squeeze()
            sub_lfs = LFdivide(data, patch_size, patch_size // 2)


            label = label.squeeze()
            label_patch_size = patch_size * args.upfactor
            sub_labels = LFdivide(label, label_patch_size, label_patch_size // 2) 


            n1, n2, u, v, c, h, w = sub_lfs.shape
            sub_lfs =  rearrange(sub_lfs, 'n1 n2 u v c h w -> (n1 n2) u v c h w')
            sub_labels = rearrange(sub_labels, 'n1 n2 u v c h w -> (n1 n2) u v c h w')  
            mini_batch = args.minibatch_test
            num_inference = (n1 * n2) // mini_batch
            with torch.no_grad():
                out_lfs = []
                for idx_inference in range(num_inference):
                    torch.cuda.empty_cache()
                    input_lfs = sub_lfs[idx_inference * mini_batch : (idx_inference+1) * mini_batch, :, :, :, :, :]
                    input_labels = sub_labels[idx_inference * mini_batch : (idx_inference+1) * mini_batch, :, :, :, :, :]  # 分块后的label输入
                    lf_out = net((input_lfs.to(args.device),input_labels.to(args.device),diffusion))
                    out_lfs.append(lf_out)
                if (n1 * n2) % mini_batch:
                    torch.cuda.empty_cache()
                    input_lfs = sub_lfs[num_inference * mini_batch :, :, :, :, :, :]
                    input_labels = sub_labels[num_inference * mini_batch :, :, :, :, :, :]
                    lf_out = net((input_lfs.to(args.device),input_labels.to(args.device),diffusion))
                    out_lfs.append(lf_out)

            out_lfs = torch.cat(out_lfs, dim=0)
            out_lfs = rearrange(out_lfs, '(n1 n2) u v c h w -> n1 n2 u v c h w', n1=n1, n2=n2)
            outLF = LFintegrate(out_lfs, patch_size * args.upfactor, patch_size * args.upfactor // 2)
            outLF = outLF[:, :, :, 0 : data.shape[3] * args.upfactor, 0 : data.shape[4] * args.upfactor]

        # print(label.max())
        # print(outLF.max())

        psnr, ssim = cal_metrics(label.squeeze(), outLF)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test


def augmentation(x, y):
    if random.random() < 0.5:  # flip along U-H direction
        x = torch.flip(x, dims=[1, 4])
        y = torch.flip(y, dims=[1, 4])
    if random.random() < 0.5:  # flip along W-V direction
        x = torch.flip(x, dims=[2, 5])
        y = torch.flip(y, dims=[2, 5])
    if random.random() < 0.5: # transpose between U-V and H-W
        x = x.permute(0, 2, 1, 3, 5, 4)
        y = y.permute(0, 2, 1, 3, 5, 4)

    "random color shuffling"
    if random.random() < 0.5:
        color = [0, 1, 2]
        random.shuffle(color)
        x, y = x[:, :, :, color, :, :], y[:, :, :, color, :, :]

    return x, y



def cosine_similarity_loss(X):
    """
    计算 (b, n, c) 张量中每批次n个c维向量之间的余弦相似度损失。
    这里假设损失是使所有向量对的相似度最大化 (即余弦相似度趋近于1)。
    """
    b, n, c = X.shape

    # 1. 对每个向量进行L2归一化
    # F.normalize 默认沿着最后一维进行归一化
    X_normalized = F.normalize(X, p=2, dim=-1)

    # 2. 计算所有向量对的点积（余弦相似度）
    # 对于每个批次，执行 (n, c) @ (c, n) -> (n, n) 矩阵乘法
    # torch.einsum 是一个非常灵活的工具，这里也可以用它来计算批次内的点积
    # 或者直接使用 @ 操作符进行批次矩阵乘法
    # (b, n, c) @ (b, c, n) -> (b, n, n)
    similarity_matrix = torch.bmm(X_normalized, X_normalized.transpose(1, 2))

    # 3. 定义损失函数
    # 我们想让相似度趋近于1，所以取负值，最小化负值就是最大化原值
    # 移除对角线元素（因为自己和自己的相似度总是1）
    # 创建一个对角线为0的掩码，用于排除自相似度
    mask = torch.eye(n, dtype=torch.bool, device=X.device).unsqueeze(0) # (1, n, n)
    mask = mask.repeat(b, 1, 1)
    
    # 得到非对角线元素
    non_diagonal_similarities = similarity_matrix[~mask].view(b, -1)

    # 我们可以计算平均余弦相似度作为损失，并取负值进行最小化
    # 目标是使余弦相似度最大化（接近1），所以损失是负的平均相似度
    loss = -torch.mean(non_diagonal_similarities)

    return loss

# # 计算损失
# loss_value = cosine_similarity_loss(input_tensor)
# print(f"余弦相似度损失: {loss_value.item()}")


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')    #设置PyTorch多进程的启动方式
    train(args)
