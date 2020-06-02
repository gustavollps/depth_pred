# import fix
# fix.fix()

from torch.utils.data import DataLoader
from cnn.cnn_mult import *
import time
import pickle
import copy
from dataset_class import CustomDataset

with open('dataset_list.pickle', 'rb') as dataset_obj:
    dataset_list = pickle.load(dataset_obj)

dataset = CustomDataset(dataset_list)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getDataLoader(batch_size, imgs_obj):
    return DataLoader(imgs_obj, batch_size=batch_size, shuffle=True)


img_loader = getDataLoader(4, dataset)
temp = next(iter(img_loader))


def adjust_learning_rate(optimizer, epoch, initial):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial * (0.5 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def trainNet(net, batch_size, n_epochs, learning_rate):
    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    # Create our loss and optimizer functions
    loss = losses.BerHuLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9,
                          weight_decay=1e-4)

    # img_loader = getDataLoader(batch_size, dataset)
    n_batches = len(img_loader)

    minimum_loss = 10000.0
    best_model = copy.deepcopy(net)
    # Time for printing
    training_start_time = time.time()

    current_loss = 0.0

    # Loop for n_epochs
    start_time = time.time()
    total_train_loss = 0
    for epoch in range(n_epochs):
        """
        if minimum_loss > current_loss > 0:
            best_model = copy.deepcopy(net)
            minimum_loss = current_loss
            torch.save(net, 'test1')
            print('Model saved!')
        """
        data = temp
        i = epoch
        # print("Epoch", epoch)
        # print("Batch", i, '/', n_batches)
        # Get inputs
        inputs, labels, mask = data
        # labels = next(iter(dep_loader))
        # Wrap them in a Variable object
        inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
        adjust_learning_rate(optimizer, epoch, learning_rate)
        # Set the parameter gradients to zero
        optimizer.zero_grad()

        # Forward pass, backward pass, optimize
        outputs = net(inputs)
        loss_size = loss(outputs, labels, mask)
        loss_size.backward()
        optimizer.step()

        # Print statistics
        current_loss = loss_size.data
        total_train_loss += loss_size.data
        print("Current: {:.2f}\tTotal: {:.2f}\tEpoch: {:d}".format(current_loss, total_train_loss, epoch))

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    print("Best model got loss of", minimum_loss)
    return best_model


from cnn.resnet_new import resnet50

net = resnet50()
net.to(device)
trainNet(net, batch_size=4, n_epochs=100, learning_rate=0.01)

torch.save(net, 'test1')
