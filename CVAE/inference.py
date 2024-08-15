from cvae import ConditionalVAE
import numpy as np
import argparse
import random
import pickle
import torch
import utils
import os
import torch.onnx

parser = argparse.ArgumentParser(description='VR Avatar')
parser.add_argument('--filename', type=str, default='checkpoint-20', help='Checkpoint file name')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--num-features', type=int, default=33, help='Number of input features')
parser.add_argument('--latent-dim', type=int, default=10, help='Dimension of latent space')
parser.add_argument('--num-hidden-units', type=int, default=128, help='Number of hidden units')
parser.add_argument('--sequence-length', type=int, default=32, help='Sequence length')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
FLAGS = parser.parse_args()


def predict(model, dataloader, args):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            if args.cuda:
                inputs = inputs.cuda()
            latent = torch.randn(len(inputs), args.latent_dim, args.sequence_length)
            preds = model.decode(latent.cuda(), inputs.cuda())
            all_preds.append(preds.cpu().numpy().transpose(0, 2, 1).reshape(len(inputs) * args.sequence_length, args.num_features))
    print("Inference completed!")
    all_preds = np.concatenate(all_preds, axis=0)

    with open(os.path.join(args.dataset_path, "aux_data"), 'rb') as handle:
        aux_data = pickle.load(handle)
    all_preds *= aux_data['labels_a']
    all_preds += aux_data['labels_b']
    np.savetxt(os.path.join(args.results_path, 'preds.txt'), all_preds, fmt='%.6f', delimiter=',', newline='\n')


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
    args.dataset_path = os.path.join(os.getcwd(), 'dataset')
    args.model_path = os.path.join(os.getcwd(), 'checkpoints')
    args.results_path = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    if not os.path.exists(args.model_path):
        raise Exception("Model not found!")
    if not os.path.exists(args.dataset_path):
        raise Exception("Dataset not found!")
    # Model
    filepath = os.path.join(args.model_path, args.filename)
    if not os.path.exists(filepath):
        raise Exception("Model not found!")
    checkpoint = torch.load(filepath)
    model = ConditionalVAE(args)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    print("Model '{}' loaded with success!".format(args.filename))
    if args.cuda:
        model.cuda()

    #onnx
    # Get the latest opset version

    device = 'cuda' if args.cuda else 'cpu'
    dummy_input_x = torch.randn(args.batch_size, args.num_features, args.sequence_length, device=device)
    dummy_input_c = torch.randn(args.batch_size, args.num_features, args.sequence_length, device=device)
    # torch.onnx.export(model, args.batch_size, "cvae.onnx", export_params=True, opset_version=10)
    torch.onnx.export(model, (dummy_input_x, dummy_input_c), "cvae.onnx", export_params=True)
    # Inference
    testloader = utils.load_data(args, False)
    predict(model, testloader, args)



if __name__ == '__main__':
    main(FLAGS)
