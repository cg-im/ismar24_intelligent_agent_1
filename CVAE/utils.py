from avatar_dataset import AvatarDataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import pickle
import os


def process_data(args):
    feature_columns = ['pelvis_aX_A', 'pelvis_aY_A', 'pelvis_aZ_A',
                       'thoraX_A_aX_A', 'thoraX_A_aY_A', 'thoraX_A_aZ_A',
                       'head_aX_A', 'head_aY_A', 'head_aZ_A',
                       'lclavicle_aX_A', 'lclavicle_aY_A', 'lclavicle_aZ_A',
                       'lhumerus_aX_A', 'lhumerus_aY_A', 'lhumerus_aZ_A',
                       'lradius_aX_A', 'lradius_aY_A', 'lradius_aZ_A',
                       'lhand_aX_A', 'lhand_aY_A', 'lhand_aZ_A',
                       'rclavicle_aX_A', 'rclavicle_aY_A', 'rclavicle_aZ_A',
                       'rhumerus_aX_A', 'rhumerus_aY_A', 'rhumerus_aZ_A',
                       'rradius_aX_A', 'rradius_aY_A', 'rradius_aZ_A',
                       'rhand_aX_A', 'rhand_aY_A', 'rhand_aZ_A']
    label_columns = ['pelvis_aX_B', 'pelvis_aY_B', 'pelvis_aZ_B',
                     'thoraX_B_aX_B', 'thoraX_B_aY_B', 'thoraX_B_aZ_B',
                     'head_aX_B', 'head_aY_B', 'head_aZ_B',
                     'lclavicle_aX_B', 'lclavicle_aY_B', 'lclavicle_aZ_B',
                     'lhumerus_aX_B', 'lhumerus_aY_B', 'lhumerus_aZ_B',
                     'lradius_aX_B', 'lradius_aY_B', 'lradius_aZ_B',
                     'lhand_aX_B', 'lhand_aY_B', 'lhand_aZ_B',
                     'rclavicle_aX_B', 'rclavicle_aY_B', 'rclavicle_aZ_B',
                     'rhumerus_aX_B', 'rhumerus_aY_B', 'rhumerus_aZ_B',
                     'rradius_aX_B', 'rradius_aY_B', 'rradius_aZ_B',
                     'rhand_aX_B', 'rhand_aY_B', 'rhand_aZ_B']

    features = []
    labels = []
    for i in range(1, args.num_partitions + 1):
        data = pd.read_csv(os.path.join(args.data_path, args.filename + '_{:d}.csv'.format(i)))
        features.append(data[feature_columns].values)
        labels.append(data[label_columns].values)
    features = np.concatenate(features, axis=0).astype(np.float32)
    labels = np.concatenate(labels, axis=0).astype(np.float32)
    assert np.isnan(features).sum() + np.isnan(labels).sum() == 0

    features_b = features.mean(0)
    features -= features_b
    features_a = features.std(0)
    features_a[[10, 20, 22, 32]] = 1.0
    features /= features_a
    labels_b = labels.mean(0)
    labels -= labels_b
    labels_a = labels.std(0)
    labels_a[[10, 15, 20, 22, 32]] = 1.0
    labels /= labels_a

    num_train_samples = round(features.shape[0] * args.train_ratio)
    train_features = features[:num_train_samples]
    train_labels = labels[:num_train_samples]
    test_features = features[num_train_samples:]
    test_labels = labels[num_train_samples:]
    train_inputs = []
    train_outputs = []
    test_inputs = []
    test_outputs = []
    for t in range(0, len(train_features) - args.sequence_length + 1, args.hop_length_train):
        train_inputs.append(train_features[t:t + args.sequence_length])
        train_outputs.append(train_labels[t:t + args.sequence_length])
    for t in range(0, len(test_features) - args.sequence_length + 1, args.hop_length_test):
        test_inputs.append(test_features[t:t + args.sequence_length])
        test_outputs.append(test_labels[t:t + args.sequence_length])
    train_inputs = np.stack(train_inputs, axis=0).transpose(0, 2, 1)
    train_outputs = np.stack(train_outputs, axis=0).transpose(0, 2, 1)
    test_inputs = np.stack(test_inputs, axis=0).transpose(0, 2, 1)
    test_outputs = np.stack(test_outputs, axis=0).transpose(0, 2, 1)

    print("Training inputs shape: ", train_inputs.shape)
    print("Training outputs shape: ", train_outputs.shape)
    print("Testing inputs shape: ", test_inputs.shape)
    print("Testing outputs shape: ", test_outputs.shape)
    np.save(os.path.join(args.dataset_path, "inputs_train"), train_inputs)
    np.save(os.path.join(args.dataset_path, "outputs_train"), train_outputs)
    np.save(os.path.join(args.dataset_path, "inputs_test"), test_inputs)
    np.save(os.path.join(args.dataset_path, "outputs_test"), test_outputs)

    aux_data = {}
    aux_data['features_a'] = features_a
    aux_data['features_b'] = features_b
    aux_data['labels_a'] = labels_a
    aux_data['labels_b'] = labels_b
    with open(os.path.join(args.dataset_path, "aux_data"), 'wb') as handle:
        pickle.dump(aux_data, handle)


def load_data(args, train=True):
    kwargs = {'num_workers': 16, 'pin_memory': True} if args.cuda else {}
    dataset = AvatarDataset(args.dataset_path, train)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=train, **kwargs)
