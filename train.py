from torch.utils.data import DataLoader
from cnn.cnn_mult import *
import time
import pickle
import copy
import losses
from datasets.nyudepthv2 import NYUDepthV2
from datasets.make3d import Make3D
from cnn import BTS
import cv2
from cnn.resnet_new import resnet50
from cnn.inception_resnet_v1 import InceptionResNetV1
from cnn.inception_resnet_v2 import InceptionResNetV2
from torch.optim.lr_scheduler import StepLR
import os
from test_model import testNet
from draw import plot2cv2, draw_debug, draw_tensor
from utils import restore_checkpoint, save_checkpoint

with open('dataset_list.pickle', 'rb') as dataset_obj:
    dataset_list = pickle.load(dataset_obj)

restore_check = False
masked = True
shuffle = True
test_on_training = True

dataset = Make3D(masked=masked, data_augmentation=15)
#dataset = NYUDepthV2(shrink_factor=10)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getDataLoader(batch_size, imgs_obj, shuffle_opt):
    return DataLoader(imgs_obj, batch_size=batch_size, shuffle=shuffle_opt)


def trainNet(net, batch_size, n_epochs, learning_rate, last_epoch_loss, last_loss, optimizer, scheduler, save=True):
    # Print all of the hyperparameters of the training iteration:
    efective_batch = batch_size
    batch_size = 2
    print("===== TRAINING SETUP =====")
    print("batch_size:", efective_batch)
    print("epochs:", n_epochs)
    print("learning rate: {:.6f}".format(learning_rate))
    print("loaded loss= {:.6f}".format(last_loss))
    print("training set size: {:d}".format(len(dataset)))
    print("=" * len("===== TRAINING SETUP ====="))
    img_loader = getDataLoader(batch_size, dataset, shuffle)
    net.train()

    learning_log = []
    test_log = []
    training_log = []

    loss = losses.SilogLoss()

    n_batches = len(img_loader)

    training_start_time = time.time()

    total_train_loss = 0
    best_loss = 9999999
    old_test_loss = last_loss
    current_step = 0


    # Loop for n_epochs
    for epoch in range(n_epochs):

        epoch_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()

        for i, data in enumerate(img_loader, 0):

            inputs, labels, mask = data

            inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
            outputs = net(inputs.float())

            draw_debug(inputs, labels, mask, outputs)

            loss_size = loss(outputs, labels, lim=((1 / 81), (70 / 81))) / (efective_batch/batch_size)

            loss_size.backward()

            if current_step % (efective_batch/batch_size) == 0:
                optimizer.zero_grad()
                optimizer.step()

            total_train_loss += loss_size.data
            epoch_loss += loss_size.data

            if (i + 1) % (print_every + 1) == 0:
                for param_group in optimizer.param_groups:
                    opt_rate = param_group['lr']
                print(
                    "Epoch {}, {:d}%".format(epoch, int(100 * (i + 1) / n_batches)))
                if test_on_training:
                    net.eval()
                    test_loss = testNet(net)
                    net.train()
                    if save is True:
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': net.state_dict(),
                            'loss': epoch_loss,
                            'test_loss': test_loss,
                            'scheduler': scheduler.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'masked': masked
                        }, test_loss, masked=masked)
                    # training_log.append(epoch_loss)
                    test_log.append(test_loss)
                    opt_rate = 0
                    for param_group in optimizer.param_groups:
                        opt_rate = param_group['lr']
                    # learning_log.append(opt_rate)
                    current_step += 1

        # net.eval()
        # test_loss = testNet(net, debug=True)
        # net.train()
        test_loss = epoch_loss
        epoch_loss = epoch_loss / len(img_loader)
        opt_rate = 0
        for param_group in optimizer.param_groups:
            opt_rate = param_group['lr']
        print(
            "Epoch {:d}\t Loss: {:f} \tTest loss: {:f}\t LR: {:f}\tTime: {:.2f}min".format(epoch,
                                                                                           epoch_loss,
                                                                                           test_loss,
                                                                                           opt_rate,
                                                                                           (
                                                                                                   time.time() - start_time) / 60)
        )
        training_log.append(epoch_loss)
        test_log.append(test_loss)
        learning_log.append(opt_rate)
        plot = plot2cv2([training_log, test_log, learning_log])
        cv2.imshow('loss', plot)
        if epoch_loss > old_test_loss:
            scheduler.step()

        old_test_loss = epoch_loss

        if save is True:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'loss': epoch_loss,
                'test_loss': test_loss,
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'masked': masked
            }, test_loss, masked=masked)

    print("Training finished, took {:.4f}s".format(time.time() - training_start_time))


net = BTS.bts_model().float()
# net = resnet50()
net.to(device)
lr = 0.001
# optimizer = optim.Adam(net.parameters(), lr=lr)
optimizer = torch.optim.Adam([{'params': net.encoder.parameters(), 'weight_decay': 1e-2},
                              {'params': net.decoder.parameters(), 'weight_decay': 0}],
                             lr=1e-4, eps=1e-6)

scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
test_loss = 999999
epoch_loss = 999999

if restore_check is True:
    net, optimizer, scheduler, epoch_loss, test_loss = restore_checkpoint(net, optimizer, scheduler, masked,
                                                                          recent=True, inception=False)

for param in optimizer.param_groups:
    lr = param['lr']

trainNet(net, batch_size=8, n_epochs=100, learning_rate=lr,
         last_epoch_loss=epoch_loss, last_loss=test_loss, optimizer=optimizer,
         scheduler=scheduler, save=True)
