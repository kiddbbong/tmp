from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models.syncnet import SyncNet_color as SyncNet
# from models import Wav2Lip, Wav2Lip_disc_qual
from models.wav2lip import Wav2Lip, Wav2Lip_disc_qual, Wav2Lip_disc_qual_full
import audio

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import warnings


os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"

writer = SummaryWriter()
now = datetime.now()
warnings.filterwarnings(action='ignore')
start_time = time.time()

parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model WITH the visual quality discriminator')

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default='./ordered_checkpoint_disc_scratch_batch16_discWEIGHT7HALF4/', type=str)
parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", default='/home/sungjoon/dataset/lrw_preprocessed/', type=str) # data dir
parser.add_argument('--syncnet_checkpoint_path', help='Load the pre-trained Expert discriminator', default='./pretrained/syncnet_004500000.pth', type=str) # lrw syncnet load
# parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default='./pretrained/checkpoint_step000100000.pth', type=str) # lrw dataset wav2lip nogan load
# parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default='./pretrained/wav2lip_gan.pth', type=str) # lrs2 dataset wav2lip load
# parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default='./ordered_checkpoint_disc_scratch/checkpoint_step000120000.pth', type=str) # lrw dataset wav2lip nogan load
parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default='./ordered_checkpoint_disc_scratch_batch16_discWEIGHT7HALF4/checkpoint_step000130000.pth', type=str) # lrw dataset wav2lip nogan load
parser.add_argument('--disc_checkpoint_path', help='Resume quality disc from this checkpoint', default=None, type=str) # lrs2 dataset disciminator load
#parser.add_argument('--disc_checkpoint_path', help='Resume quality disc from this checkpoint', default='./pretrained/visual_quality_disc.pth', type=str) # lrs2 dataset disciminator load
# parser.add_argument('--disc_checkpoint_path', help='Resume quality disc from this checkpoint', default='./checkpoint/disc_checkpoint_step000220000.pth', type=str) # lrs2 dataset disciminator load

args = parser.parse_args()

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16

class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)
        self.split = split
        print(split, len(self.all_videos))

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)

        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '%05d.jpg'%(frame_id))

            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]
        

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        # start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing # Original
        start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):

        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:

            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(args.data_root, self.split, vidname, '*.jpg')))

            if len(img_names) <= 3 * syncnet_T:

                continue
            
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:

                wrong_img_name = random.choice(img_names)


            window_fnames = self.get_window(img_name)

            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:

                continue

            window = self.read_window(window_fnames)
            if window is None:

                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:

                continue

            try:
                wavpath = join(args.data_root, self.split, vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)
                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)
            
            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None: continue

            window = self.prepare_window(window)
            y = window.copy()
            window[:, :, window.shape[2]//2:] = 0.

            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(y)

            return x, indiv_mels, mel, y

def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()
    elif isinstance(submodule, nn.Linear):
        torch.nn.init.xavier_uniform(submodule.weight)
        submodule.bias.data.fill_(0.01)

def save_sample_images(x, g, gt, global_step, checkpoint_dir, split):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "{}_samples_step{:09d}".format(split, global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])

def save_sample_with_rot_images(x, g, gt, rot_g, rot_gt, global_step, checkpoint_dir, split):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    
    if rot_g != None and rot_gt != None:
        rot_g = (rot_g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
        rot_gt = (rot_gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "{}_samples_step{:09d}".format(split, global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    if rot_g.all() != None and rot_gt.all() != None:
        collage = np.concatenate((refs, inps, g, gt, rot_g, rot_gt), axis=-2)
    else:
        collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/rot_{}_{}.jpg'.format(folder, batch_idx, t), c[t])

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

device = torch.device("cuda" if use_cuda else "cpu")
syncnet = SyncNet().to(device)
for p in syncnet.parameters():
    p.requires_grad = False

recon_loss = nn.L1Loss()
def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)

def rotate_batchs(g, gt):
    if g.shape[-1] != hparams.img_size or g.shape[-2] != hparams.img_size:
        print(" Error for Rotate Fake and Real Shape ! ")
        return g, gt
    if g.shape != gt.shape:
        print(" Error for Different Shape ! ")
        return g, gt

    for i in range(g.shape[0]):
        idx = np.random.randint(4)
        g_one = g[i]
        gt_one = gt[i]
        rot_g = torch.rot90(g_one, idx, [2, 3])
        rot_gt = torch.rot90(gt_one, idx, [2, 3])
        rot_g = rot_g[None, :]
        rot_gt = rot_gt[None, :]
        if i == 0:
            batch_g = rot_g
            batch_gt = rot_gt
        else:
            batch_g = torch.cat((batch_g, rot_g), 0)
            batch_gt = torch.cat((batch_gt, rot_gt), 0)

    return batch_g, batch_gt

def train(device, model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    resumed_step = global_step

    # global_step = 0
    # global_epoch = 0

    print(f" [In Train] Global Step --- {global_step} / Global Epoch --- {global_epoch} ")

    avg_total_loss = []
    avg_l1_loss = []
    avg_sync_loss = []
    avg_perceptual_loss = []
    avg_fake_loss = []
    avg_real_loss = []
    ####### Merge Disc Loss
    avg_disc_loss = []
    avg_pred_gt = []
    avg_pred_g = []

    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))

        running_sync_loss, running_l1_loss, disc_loss, running_perceptual_loss = 0., 0., 0., 0.
        running_disc_real_loss, running_disc_fake_loss = 0., 0.
        # prog_bar = tqdm(enumerate(train_data_loader))

        for step, (x, indiv_mels, mel, gt) in enumerate(train_data_loader):

            disc.train()
            model.train()

            x = x.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)

            ### Train generator now. Remove ALL grads. 
            optimizer.zero_grad()
            disc_optimizer.zero_grad()

            g = model(indiv_mels, x)

            if hparams.syncnet_wt > 0.:
                sync_loss = get_sync_loss(mel, g)
            else:
                sync_loss = 0.

            if hparams.disc_wt > 0.:
                # perceptual_loss = disc.perceptual_forward(g) # Original Not DataParallel
                perceptual_loss = disc.module.perceptual_forward(g) # For DataParallel
            else:
                perceptual_loss = 0.

            l1loss = recon_loss(g, gt)

            loss = hparams.syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss + \
                                    (1. - hparams.syncnet_wt - hparams.disc_wt) * l1loss

            loss.backward()
            optimizer.step()

            avg_total_loss.append(loss.item())
            ### Remove all gradients before Training disc
            disc_optimizer.zero_grad()

            rot_g, rot_gt = None, None

            if hparams.disc_wt > 0.:
                rot_g, rot_gt = rotate_batchs(g, gt)

                pred = disc(gt) # No Rotate
                # pred = disc(rot_gt)
                disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))
                disc_real_loss.backward()
                pred_gt = torch.mean(pred)

                pred = disc(g.detach()) # No Rotate
                # pred = disc(rot_g.detach())
                disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))
                disc_fake_loss.backward()
                pred_g = torch.mean(pred)

                #### Merge Disc Loss
                disc_loss = hparams.disc_wt * 0.5 * (disc_real_loss + disc_fake_loss)

                # disc_loss.backward()

                disc_optimizer.step()

                avg_real_loss.append(disc_real_loss.item())
                avg_fake_loss.append(disc_fake_loss.item())
                
                #### Merge Disc Loss
                avg_disc_loss.append(disc_loss.item())
                avg_pred_gt.append(pred_gt.item())
                avg_pred_g.append(pred_g.item())

            else:
                disc_real_loss = 0.
                disc_fake_loss = 0.

                avg_real_loss = 0.
                avg_fake_loss = 0.

                #### Merge Disc Loss
                avg_disc_loss = hparams.disc_wt * 0.5 * (disc_real_loss + disc_fake_loss)
                avg_pred_gt = 0.
                avg_pred_g = 0.


            #### SJ Change Later ####
            # avg_real_loss.append(disc_real_loss.item())
            # avg_fake_loss.append(disc_fake_loss.item())

            # Logs
            global_step += 1
            cur_session_steps = global_step - resumed_step

            avg_l1_loss.append(l1loss.item())

            if hparams.syncnet_wt > 0.:
                avg_sync_loss.append(sync_loss.item())


            if hparams.disc_wt > 0.:
                avg_perceptual_loss.append(perceptual_loss.item())

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch, prefix='')
                save_checkpoint(disc, disc_optimizer, global_step, checkpoint_dir, global_epoch, prefix='disc_')
            
            if global_step % hparams.eval_interval == 0:
                save_sample_images(x, g, gt, global_step, checkpoint_dir, 'train')
                # save_sample_with_rot_images(x, g, gt, rot_g, rot_gt, global_step, checkpoint_dir, 'train')
                with torch.no_grad():
                    average_sync_loss = eval_model(test_data_loader, global_step, device, model, disc)
                    if average_sync_loss < .3:
                        hparams.set_hparam('syncnet_wt', 0.03)
                        hparams.set_hparam('disc_wt', 0.07)
                    elif average_sync_loss < .75:
                        hparams.set_hparam('syncnet_wt', 0.01)

            if global_step%100 == 0:
                t = time.time()-start_time

                mean_l1_loss = np.mean(avg_l1_loss)

                if len(avg_sync_loss)==0:
                    mean_sync_loss = 0
                else:
                    mean_sync_loss = np.sum(avg_sync_loss)/len(avg_sync_loss)

                if len(avg_perceptual_loss)==0:
                    mean_perceptual_loss = 0
                else:
                    mean_perceptual_loss = np.sum(avg_perceptual_loss)/len(avg_perceptual_loss)

                mean_fake_loss = np.mean(avg_fake_loss)
                mean_real_loss = np.mean(avg_real_loss)
                ###### Merge Disc Loss
                mean_disc_loss = np.mean(avg_disc_loss)
                mean_pred_gt = np.mean(avg_pred_gt)
                mean_pred_g = np.mean(avg_pred_g)

                total_loss = np.mean(avg_total_loss)

                # print('Step: {}, Total Loss: {}, L1 loss: {}, Sync loss: {}, Perceptual loss: {}, Fake loss: {}, Real loss: {}, Time: {}'
                #       .format(global_step, total_loss, mean_l1_loss, mean_sync_loss, mean_perceptual_loss, mean_fake_loss, mean_real_loss, t))
                print('Step: {}, Total Loss: {}, L1 loss: {}, Sync loss: {}, Perceptual loss: {}, Fake loss: {}, Real loss: {}, Disc loss: {}, Time: {}'
                      .format(global_step, total_loss, mean_l1_loss, mean_sync_loss, mean_perceptual_loss, mean_fake_loss, mean_real_loss, mean_disc_loss, t))

                writer.add_scalar("Total loss", total_loss, global_step)
                writer.add_scalar("L1 loss", mean_l1_loss, global_step)
                writer.add_scalar("Sync loss", mean_sync_loss, global_step)
                writer.add_scalar("Perceptual loss", mean_perceptual_loss, global_step)
                writer.add_scalar("Fake loss", mean_fake_loss, global_step)
                writer.add_scalar("Real loss", mean_real_loss, global_step)
                ###### Merge Disc Loss
                writer.add_scalar("Discrminator loss", mean_disc_loss, global_step)
                writer.add_scalar("Prediction GT", mean_pred_gt, global_step)
                writer.add_scalar("Prediction Generator", mean_pred_g, global_step)
                

                with open("./logs/%s_train.txt"%(now.strftime('%m_%d_%H-%M-%S')), "a") as f:
                    f.write('%08d\t%7.7f\t%7.7f\t%7.7f\t%7.7f\t%7.7f\t%7.7f\t%7.7f\n'
                            % (global_step, total_loss, mean_l1_loss, mean_sync_loss, mean_perceptual_loss, mean_fake_loss, mean_real_loss, t))

                avg_l1_loss = []
                avg_sync_loss = []
                avg_perceptual_loss = []
                avg_fake_loss = []
                avg_real_loss = []
                avg_total_loss= []
                ###### Merge Disc Loss
                avg_disc_loss = []
                avg_pred_gt = []
                avg_pred_g = []

        global_epoch += 1

def eval_model(test_data_loader, global_step, device, model, disc):
    eval_steps = 300
    print('Evaluating for {} steps'.format(eval_steps))

    avg_total_loss = []
    avg_l1_loss = []
    avg_sync_loss = []
    avg_perceptual_loss = []
    avg_fake_loss = []
    avg_real_loss = []
    ###### Merge Disc Loss
    avg_disc_loss = []
    avg_pred_gt = []
    avg_pred_g = []

    number = random.randrange(0, eval_steps)
    for step, (x, indiv_mels, mel, gt) in enumerate((test_data_loader)):

        model.eval()
        disc.eval()

        x = x.to(device)
        mel = mel.to(device)
        indiv_mels = indiv_mels.to(device)
        gt = gt.to(device)

        pred = disc(gt)
        disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))
        pred_gt = torch.mean(pred)

        g = model(indiv_mels, x)
        pred = disc(g)
        disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))
        pred_g = torch.mean(pred)

        ###### Merge Disc Loss
        disc_loss = 0.5 * (disc_real_loss + disc_fake_loss)
        avg_disc_loss.append(disc_loss.item())
        avg_pred_gt.append(pred_gt.item())
        avg_pred_g.append(pred_g.item())


        avg_real_loss.append(disc_real_loss.item())
        avg_fake_loss.append(disc_fake_loss.item())

        sync_loss = get_sync_loss(mel, g)

        if hparams.disc_wt > 0.:
            # perceptual_loss = disc.perceptual_forward(g)
            perceptual_loss = disc.module.perceptual_forward(g)
        else:
            perceptual_loss = 0.

        l1loss = recon_loss(g, gt)

        loss = hparams.syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss + \
                                (1. - hparams.syncnet_wt - hparams.disc_wt) * l1loss

        avg_total_loss.append(loss.item())
        avg_l1_loss.append(l1loss.item())
        avg_sync_loss.append(sync_loss.item())

        if hparams.disc_wt > 0.:
            avg_perceptual_loss.append(perceptual_loss.item())

        if step == number:
            save_sample_images(x, g, gt, global_step, checkpoint_dir, 'val')
        if step > eval_steps: break

    mean_l1_loss = np.mean(avg_l1_loss)
    if len(avg_sync_loss) == 0:
        mean_sync_loss = 0
    else:
        mean_sync_loss = np.sum(avg_sync_loss) / len(avg_sync_loss)
    if len(avg_perceptual_loss) == 0:
        mean_perceptual_loss = 0
    else:
        mean_perceptual_loss = np.sum(avg_perceptual_loss) / len(avg_perceptual_loss)
    mean_fake_loss = np.mean(avg_fake_loss)
    mean_real_loss = np.mean(avg_real_loss)
    total_loss = np.mean(avg_total_loss)

    ###### Merge Disc Loss
    mean_disc_loss = np.mean(avg_disc_loss)
    mean_pred_gt = np.mean(avg_pred_gt)
    mean_pred_g = np.mean(avg_pred_g)

    writer.add_scalar("val Total loss", total_loss, global_step)
    writer.add_scalar("val L1 loss", mean_l1_loss, global_step)
    writer.add_scalar("val Sync loss", mean_sync_loss, global_step)
    writer.add_scalar("val Perceptual loss", mean_perceptual_loss, global_step)
    writer.add_scalar("val Fake loss", mean_fake_loss, global_step)
    writer.add_scalar("val Real loss", mean_real_loss, global_step)
    ###### Merge Disc Loss
    writer.add_scalar("val Disc loss", mean_disc_loss, global_step)
    writer.add_scalar("val Prediction GT", mean_pred_gt, global_step)
    writer.add_scalar("val Prediction Generator", mean_pred_g, global_step)

    print(
        '### Validation Step: {}, Total Loss: {}, L1 loss: {}, Sync loss: {}, Perceptual loss: {}, Fake loss: {}, Real loss: {}'
        .format(global_step, total_loss, mean_l1_loss, mean_sync_loss, mean_perceptual_loss, mean_fake_loss,
                mean_real_loss))

    with open("./logs/%s_val.txt" % (now.strftime('%m_%d_%H-%M-%S')), "a") as f:
        f.write('%08d\t%7.7f\t%7.7f\t%7.7f\t%7.7f\t%7.7f\t%7.7f\n'
                % (global_step, total_loss, mean_l1_loss, mean_sync_loss, mean_perceptual_loss, mean_fake_loss,
                   mean_real_loss))

    return mean_sync_loss


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]

    new_s = {}
    for k, v in s.items():
        # new_s[k.replace('module.', '')] = v
        new_s[k] = v

    if path==args.syncnet_checkpoint_path:
        print('syncnet strictly load')
        model.load_state_dict(new_s)
    else:
        model.load_state_dict(new_s, strict=True)
        # model.load_state_dict(new_s, strict=False)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model

if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir

    # Dataset and Dataloader setup
    train_dataset = Dataset('train')
    test_dataset = Dataset('val')

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.batch_size,
        num_workers=4)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    # model = Wav2Lip().to(device)
    # disc = Wav2Lip_disc_qual().to(device)

    # Multi GPU
    model = nn.DataParallel(Wav2Lip(), device_ids=list(range(2)))
    disc = nn.DataParallel(Wav2Lip_disc_qual(), device_ids=list(range(2)))
    # disc = nn.DataParallel(Wav2Lip_disc_qual_full(), device_ids=list(range(2)))
    model.to(device)
    disc.to(device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('total DISC trainable params {}'.format(sum(p.numel() for p in disc.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam([p for p in disc.parameters() if p.requires_grad],
                           lr=hparams.disc_initial_learning_rate, betas=(0.5, 0.999))

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=True, overwrite_global_states=True)

    if args.disc_checkpoint_path is not None:
        print(f" ######### DISC LOAD #########")
        load_checkpoint(args.disc_checkpoint_path, disc, disc_optimizer, 
                                reset_optimizer=True, overwrite_global_states=False)

    ######## Apply Disc Initializer
    else:
        print(f" !!!!!! Initialize Discriminator !!!!!!")
        disc.apply(weight_init_xavier_uniform)

    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, 
                                overwrite_global_states=False)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    print(f" [In Main] Global Step --- {global_step} / Global Epoch --- {global_epoch} ")

    # Train!
    train(device, model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs)
