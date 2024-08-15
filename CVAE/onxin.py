import onnxruntime as ort
import numpy as np
import torch
import argparse
import pickle
import os

# Argument Parsing (as you provided)
parser = argparse.ArgumentParser(description='VR Avatar')
parser.add_argument('--filename', type=str, default='checkpoint-20', help='Checkpoint file name')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--num-features', type=int, default=33, help='Number of input features')
parser.add_argument('--latent-dim', type=int, default=10, help='Dimension of latent space')
parser.add_argument('--num-hidden-units', type=int, default=128, help='Number of hidden units')
parser.add_argument('--sequence-length', type=int, default=32, help='Sequence length')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
FLAGS = parser.parse_args()


def onnx_single_sequence_inference(model_path, x_sequence, c_sequence, args):
    # Load the ONNX model
    session = ort.InferenceSession(model_path)

    # Convert the sequences to numpy array
    if args.cuda:
        x_sequence = x_sequence.cpu().numpy()
        c_sequence = c_sequence.cpu().numpy()

    # Prepare the input data for ONNX
    ort_inputs = {
        session.get_inputs()[0].name: x_sequence,
        session.get_inputs()[1].name: c_sequence
    }
    print("ORT INPUTS")
    print(ort_inputs)
    preds = session.run(None, ort_inputs)[0]
    print("-----------")

    # Post-process the Results
    #with open(os.path.join(args.dataset_path, "aux_data"), 'rb') as handle:
    #    aux_data = pickle.load(handle)
    #preds *= aux_data['labels_a']
    #preds += aux_data['labels_b']

    #print("ONNX Inference for single sequence completed!")

    # Post-process the Results
    with open(os.path.join(args.dataset_path, "aux_data"), 'rb') as handle:
        aux_data = pickle.load(handle)
    preds *= aux_data['labels_a'].reshape(1, -1, 1)
    preds += aux_data['labels_b'].reshape(1, -1, 1)

    return preds


def main(args):

    # Set CUDA availability
    args.cuda = torch.cuda.is_available()

    # Set paths (assuming dataset path is needed for aux data)
    args.dataset_path = os.path.join(os.getcwd(), 'dataset')

    # Prepare a single sequence for testing
    x_sequence = torch.randn(1, args.num_features, args.sequence_length).repeat(128, 1, 1)  # Example sequence for x
    c_sequence = torch.randn(1, args.num_features, args.sequence_length).repeat(128, 1, 1)  # Example sequence for c
    if args.cuda:
        x_sequence = x_sequence.cuda()
        c_sequence = c_sequence.cuda()

    # Inference using ONNX model for a single sequence
    onnx_model_path = "cvae.onnx"
    preds = onnx_single_sequence_inference(onnx_model_path, x_sequence, c_sequence, args)
    print(preds[0])


if __name__ == '__main__':
    main(FLAGS)
