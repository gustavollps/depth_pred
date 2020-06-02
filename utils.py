import os
import torch


def restore_checkpoint(net, optimizer, scheduler, masked, recent=False, inception=False):
    if masked is True:
        mask = 'MASKED'
    else:
        mask = ''
    if inception is True:
        mask = mask + "_INCEP"

    if recent is True:
        checkpoint_file = './checkpoints/' + mask + 'checkpoint_latest'
        checkpoint = torch.load(checkpoint_file)
        net.load_state_dict(checkpoint['state_dict'])
        test_loss = 1
        epoch_loss = 1
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            test_loss = checkpoint['test_loss']
            epoch_loss = checkpoint['loss']
        print("Restored latest " + mask + "checkpoint")
        return net, optimizer, scheduler, epoch_loss, test_loss

    files = [f for f in os.listdir('./checkpoints') if os.path.isfile(os.path.join('./checkpoints', f))]
    test_loss = 9999999
    epoch_loss = 9999999
    checkpoint_file = None
    if len(files) != 0:
        for s in files:
            if s.find('latest') == -1:
                if masked is not True and s.find('MASKED') == -1:
                    loss = float(s[s.find(':') + 1:])
                    if test_loss > loss:
                        test_loss = loss
                        checkpoint_file = "./checkpoints/" + mask + "checkpoint_best_loss:{:.6f}".format(loss)
                if masked is True and s.find('MASKED') != -1:
                    loss = float(s[s.find(':') + 1:])
                    if test_loss > loss:
                        test_loss = loss
                        checkpoint_file = "./checkpoints/" + mask + "checkpoint_best_loss:{:.6f}".format(loss)

    print("Loaded checkpoint file: " + checkpoint_file)
    if checkpoint_file is not None:
        checkpoint = torch.load(checkpoint_file)
        net.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            test_loss = checkpoint['test_loss']
            epoch_loss = checkpoint['loss']

    return net, optimizer, scheduler, epoch_loss, test_loss


def save_checkpoint(state, test_loss, inception=False, masked=True, safe=True):
    if masked is True:
        if inception:
            filename = './checkpoints/MASKED_INCEPcheckpoint'
        else:
            filename = './checkpoints/MASKED_checkpoint'
        mask = 'MASKED'
    else:
        if inception:
            filename = './checkpoints/INCEP_checkpoint'
        else:
            filename = './checkpoints/checkpoint'
        mask = ''

    # torch.save(state, filename + "_" + str(epoch) + "_" + "loss:{:.2f}".format(test_loss))
    files = [f for f in os.listdir('./checkpoints') if os.path.isfile(os.path.join('./checkpoints', f))]
    torch.save(state, filename + '_latest')
    if len(files) == 0:
        torch.save(state, filename + "_best_" + "loss:{:.6f}".format(test_loss))

    for s in files:
        if s.find('latest') == -1:
            if mask == '' and s.find('MASKED') == -1:
                loss = float(s[s.find(':') + 1:])
                if test_loss > loss:
                    print('Not the best loss')
                    return

            if mask == 'MASKED' and s.find('MASKED') != -1:
                loss = float(s[s.find(':') + 1:])
                if test_loss > loss:
                    print('Not the best loss')
                    return

    torch.save(state, filename + "_best_" + "loss:{:.6f}".format(test_loss))
    if safe is not True and len(loss) > 0:
        try:
            os.remove(filename + "_best_" + "loss:{:.6f}".format(min(loss)))
        except:
            print("No older best checkpoint")
