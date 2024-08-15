from cvae import ConditionalVAE, VAE_loss
import torch.optim as optim
import numpy as np
import argparse
import random
import torch
import utils
import os

parser = argparse.ArgumentParser(description='VR Avatar')
# Data
parser.add_argument('--filename', type=str, default='S1', help='Data file name')
parser.add_argument('--num-partitions', type=int, default=3, help='Number of data partitions')
parser.add_argument('--hop-length-train', type=int, default=8, help='Hop length for creating the training set')
parser.add_argument('--hop-length-test', type=int, default=16, help='Hop length for creating the evaluation set')
parser.add_argument('--train-ratio', type=float, default=0.8, help='Training data ratio')
# Model
parser.add_argument('--num-features', type=int, default=33, help='Number of input features')
parser.add_argument('--latent-dim', type=int, default=10, help='Dimension of latent space')
parser.add_argument('--num-hidden-units', type=int, default=128, help='Number of hidden units')
parser.add_argument('--sequence-length', type=int, default=32, help='Sequence length')
# Training
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
parser.add_argument('--num-epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay factor')
FLAGS = parser.parse_args()


def train_epoch(model, dataloader, optimizer, args):
    model.train()
    error_list = []
    loss_list = []
    for inputs, outputs in dataloader:
        if args.cuda:
            inputs, outputs = inputs.cuda(), outputs.cuda()
        preds, mean, logvar = model(outputs, inputs)
        loss = VAE_loss(preds, outputs, mean, logvar)
        loss_list.append(loss.item())
        error = torch.abs(preds - outputs).mean()
        error_list.append(error.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss = np.mean(loss_list)
    error = np.mean(error_list)
    return loss, error


def train(args, data_ready=False):
    if not data_ready:
        utils.process_data(args)
    model = ConditionalVAE(args)
    if args.cuda:
        model.cuda()
    trainloader = utils.load_data(args, True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(1, args.num_epochs + 1):
        loss_train, error_train = train_epoch(model, trainloader, optimizer, args)
        print("Epoch {:d} | Train loss: {:.5f} | Train error: {:.4f}".format(epoch, loss_train, error_train))
        info = {'epoch': epoch, 'state_dict': model.state_dict()}
        filename = "checkpoint-{:d}".format(epoch)
        filepath = os.path.join(args.model_path, filename)
        torch.save(info, filepath)


def main(args):
    # CUDA support
    args.cuda = torch.cuda.is_available()
    # Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Folders
    args.data_path = os.path.join(os.getcwd(), 'data')
    args.dataset_path = os.path.join(os.getcwd(), 'dataset')
    args.model_path = os.path.join(os.getcwd(), 'checkpoints')
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)
    if not os.path.exists(args.data_path):
        raise Exception("Data not found!")
    # Training
    train(args, data_ready=False)


if __name__ == '__main__':
    main(FLAGS)
