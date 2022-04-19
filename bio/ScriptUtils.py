import argparse
import numpy as np
import torch
import wandb
from torch.utils.data import TensorDataset
from transformers import set_seed

from Data import BiasInBiosDataLinear


def parse_training_args(replace_top_layer = False):
    parser = argparse.ArgumentParser(description='Run finetuning training process on Bias in Bios dataset.')
    parser.add_argument('--model', default='roberta', type=str, choices=['roberta-base', 'deberta-base'], help='model to use as a feature extractor')
    parser.add_argument('--batch_size', '-b', default=64, type=int, help='the batch size for the training process')
    parser.add_argument('--data', '-d', required=False, type=str, help='the data type',
                        choices=["raw", "scrubbed"], default='raw')
    parser.add_argument('--balanced', type=str, help='balancing of the data',
                        choices=["subsampled", "oversampled", "original"], default="original")
    parser.add_argument('--lr', default=5e-5, type=float, help='the learning rate')
    parser.add_argument('--epochs', '-e', default=10, type=int, help='the number of epochs')
    parser.add_argument('--seed', '-s', default=0, type=int, help='the random seed')
    parser.add_argument('--embedding_seed', default=None, type=int, help='the random seed embedding was trained with')
    parser.add_argument('--printevery', '-pe', default=1, type=int, help='print results every this number of epochs')
    parser.add_argument('--checkpointevery', '-ce', default=1, type=int, help='print results every this number of epochs')

    if replace_top_layer:
        parser.add_argument('--embedding_training_data', required=False, type=str, help='the data type of pretrained embeddings',
                        choices=["raw", "scrubbed", "name"], default='raw')
        parser.add_argument('--embedding_training_balanced', type=str, help='balancing of the data for pretrained embeddings',
                            choices=["subsampled", "oversampled", "original"], default="original")

    args = parser.parse_args()

    print("Batch size:", args.batch_size)
    print("Data type:", args.data)
    print("Balancing:", args.balanced)
    print("Learning rate:", args.lr)
    print("Number of epochs:", args.epochs)
    print("Random seed:", args.seed)
    print("Print Every:", args.printevery)
    print("Checkpoint Every:", args.checkpointevery)

    return args

def parse_testing_args():
    parser = argparse.ArgumentParser(description='Run testing on Bias in Bios dataset.')
    parser.add_argument('--batch_size', '-b', default=64, type=int, help='the batch size to test with')
    parser.add_argument('--training_data', required=False, type=str, help='the data type the model was trained on',
                        choices=["raw", "scrubbed", "name", "scrubbed_extra"], default="raw")
    parser.add_argument('--testing_data', required=False, type=str, help='the data type to test on',
                        choices=["raw", "scrubbed", "name", "scrubbed_extra"], default="raw")
    parser.add_argument('--training_balanced', type=str, help='balancing of the training data',
                        choices=["subsampled", "oversampled", "original"], default="original")
    parser.add_argument('--testing_balanced', type=str, help='balancing of the test data',
                        choices=["subsampled", "oversampled", "original"], default="original")
    parser.add_argument('--split', type=str, help='the split type to test',
                        choices=["train", "test", "valid"], default="test")
    parser.add_argument('--seed', '-s', default=0, type=int, help='the random seed')
    parser.add_argument('--model', default='roberta-base', type=str, choices=['roberta-base', 'deberta-base'], help='model to use as a feature extractor')

    args = parser.parse_args()

    print("Batch size:", args.batch_size)
    print("Training data type:", args.training_data)
    print("Testing data type:", args.testing_data)
    print("Training Balancing:", args.training_balanced)
    print("Testing Balancing:", args.testing_balanced)
    print("Split Type:", args.split)
    print("Random seed:", args.seed)

    return args

def preprocess_probing_data(X, z):
    z[z == 'F'] = 1
    z[z == 'M'] = 0
    z = z.astype(int)
    X, z = X, torch.tensor(z).long()

    return TensorDataset(X, z)

def load_probing_dataset(path):
    data = torch.load(path)

    ds = preprocess_probing_data(torch.tensor(data['X']), data['z'])
    return ds

def load_bias_in_bios_vectors(args):

    seed = args.seed
    model_seed = args.model_seed

    if args.model == "finetuning":
        data_train = BiasInBiosDataLinear(f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/{args.model}/{args.training_data}/{args.training_balanced}/seed_{model_seed}/vectors_{args.type}_{args.feature_extractor}_128.pt",
                                        seed, "train", args.testing_balanced)
        data_valid = BiasInBiosDataLinear(f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/{args.model}/{args.training_data}/{args.training_balanced}/seed_{model_seed}/vectors_{args.type}_{args.feature_extractor}_128.pt",
                                        seed, "valid", args.testing_balanced)
        data_test = BiasInBiosDataLinear(f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/{args.model}/{args.training_data}/{args.training_balanced}/seed_{model_seed}/vectors_{args.type}_{args.feature_extractor}_128.pt",
                                        seed, "test", args.testing_balanced)
    elif args.model == "coref":
        path = f"../data/biosbias/vectors_extracted_from_trained_models/{args.model}/coref-model/finetuned/{args.training_balanced}/model_{model_seed}/vectors_{args.type}_{args.feature_extractor}_128.pt"
        data_train = BiasInBiosDataLinear(
            path, seed, "train", args.testing_balanced)
        data_valid = BiasInBiosDataLinear(
            path, seed, "valid", args.testing_balanced)
        data_test = BiasInBiosDataLinear(
            path, seed, "test", args.testing_balanced)
    elif args.model == "random":
        data_train = BiasInBiosDataLinear(f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/random/seed_{seed}/vectors_{args.type}_{args.feature_extractor}_128.pt",
                                            seed, "train", args.testing_balanced)
        data_valid = BiasInBiosDataLinear(f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/random/seed_{seed}/vectors_{args.type}_{args.feature_extractor}_128.pt",
                                            seed, "valid", args.testing_balanced)
        data_test = BiasInBiosDataLinear(f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/random/seed_{seed}/vectors_{args.type}_{args.feature_extractor}_128.pt",
                                            seed, "test", args.testing_balanced)
    else:
        data_train = BiasInBiosDataLinear(f"../data/biosbias/vectors_{args.type}_{args.feature_extractor}_128.pt",
                                        seed, "train", args.testing_balanced)
        data_valid = BiasInBiosDataLinear(f"../data/biosbias/vectors_{args.type}_{args.feature_extractor}_128.pt",
                                        seed, "valid", args.testing_balanced)
        data_test = BiasInBiosDataLinear(f"../data/biosbias/vectors_{args.type}_{args.feature_extractor}_128.pt",
                                        seed, "valid", args.testing_balanced)

    return preprocess_probing_data(data_train.dataset.tensors[0], data_train.z),\
           preprocess_probing_data(data_valid.dataset.tensors[0], data_valid.z),\
           preprocess_probing_data(data_test.dataset.tensors[0], data_test.z)

def load_winobias_vectors(args, preprocess=True):
    if preprocess:
        load_fn = load_probing_dataset
    else:
        load_fn = torch.load

    if args.training_task == "coref" and args.model == "finetuned":
        data = load_fn(f"../data/winobias/extracted_vectors/{args.model}/finetuned/{args.training_balanced}/number_{args.model_number}/vectors_{args.model}.pt")
    if args.training_task == "biasinbios" and args.model == "finetuned":
        data = load_fn(f"../data/winobias/extracted_vectors/{args.model}/finetuned/bios/{args.training_data}/{args.training_balanced}/seed_{args.model_seed}/vectors_{args.model}.pt")
    if args.model == "random":
        data = load_fn(f"../data/winobias/extracted_vectors/{args.model}/random/seed_{args.seed}/vectors_{args.model}.pt")
    if args.model == "basic":
        data = load_fn(f"../data/winobias/extracted_vectors/{args.model}/basic/vectors_{args.model}.pt")

    return data, None, data

def get_avg_gap(gap):
    gap = np.array(gap)
    f = np.mean(gap[gap > 0])
    m = -np.mean(gap[gap < 0])
    return {"f": f, "m": m}

def get_gap_sum(gap):
    return np.abs(np.array(gap)).sum()

def log_test_results(res):
    wandb.run.summary[f"acc"] = res['acc']
    wandb.run.summary[f"avg_loss"] = res['loss']

    perc = res['perc']

    # gaps
    wandb.run.summary[f"tpr_gap-pearson"] = res['pearson_tpr_gap']
    wandb.run.summary[f"tpr_gap-abs_sum"] = get_gap_sum(res['tpr_gap'])
    table_data = [[x, y] for (x, y) in zip(perc, res['tpr_gap'])]
    table = wandb.Table(data=table_data, columns=["perc of females", "tpr gap"])
    wandb.log({f"tpr_gap_chart": wandb.plot.line(table, "perc of females", "tpr gap",
                                                    title=f"tpr gap chart")})

    wandb.run.summary[f"fpr_gap-pearson"] = res['pearson_fpr_gap']
    wandb.run.summary[f"fpr_gap-abs_sum"] = get_gap_sum(res['fpr_gap'])
    table_data = [[x, y] for (x, y) in zip(perc, res['fpr_gap'])]
    table = wandb.Table(data=table_data, columns=["perc of females", "fpr gap"])
    wandb.log({f"fpr_gap_chart": wandb.plot.line(table, "perc of females", "fpr gap",
                                                    title=f"fpr gap chart")})

    wandb.run.summary[f"precision_gap-pearson"] = res['pearson_precision_gap']
    wandb.run.summary[f"precision_gap-abs_sum"] = get_gap_sum(res['precision_gap'])
    table_data = [[x, y] for (x, y) in zip(perc, res['precision_gap'])]
    table = wandb.Table(data=table_data, columns=["perc of females", "precision gap"])
    wandb.log({f"precision_gap_chart": wandb.plot.line(table, "perc of females", "precision gap",
                                                    title=f"precision gap chart")})

    # Allennlp metrics

    ## independence
    wandb.run.summary['independence'] = res['independence']
    wandb.run.summary['independence-sum'] = res['independence_sum']

    ## separation
    wandb.run.summary['separation'] = res['separation']
    wandb.run.summary['separation_gap-abs_sum'] = get_gap_sum(res['separation_gaps'])

    ## sufficiency
    wandb.run.summary['sufficiency'] = res['sufficiency']
    wandb.run.summary['sufficiency_gap-abs_sum'] = get_gap_sum(res['sufficiency_gaps'])
