import torch
from tqdm import tqdm
import torch.nn as nn
import argparse
import torch.optim as optim
from data import SentimentDataset
from models import bow, lstm
import utils
import numpy as np
import random
import os
import json
import csv


# load cuda if the system has GPU
# use MPS if available, otherwise use CPU
device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def main(args):

    seeds = [1, 42, 1337]

    # load data
    train_dataset = SentimentDataset(
        split="train", lower=False, supervise_nodes=args.supervise_nodes
    )
    dev_dataset = SentimentDataset(
        split="dev", lower=False, supervise_nodes=False
    )  # keep dev and test as is
    test_dataset = SentimentDataset(
        split="test", lower=False, supervise_nodes=False
    )

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    train_epoch_losses_list = []
    val_epoch_metrics_list = []
    test_metrics_list = []
    max_epochs = []
    for seed in seeds:

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # initialize model
        if args.model == "BOW":
            vocab_size = len(train_dataset.vocab.w2i)
            n_classes = len(train_dataset.vocab.t2i)
            model = bow.BOW(vocab_size, n_classes, train_dataset.vocab)
            model = model.to(device)
            print(model)
            utils.print_parameters(model)
            optimizer = optim.Adam(
                model.parameters(),
                lr=0.01,
                weight_decay=0.0001,
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
            prep_fn = utils.prepare_minibatch
        elif args.model == "CBOW":
            embedding_dim = 300
            n_classes = len(train_dataset.vocab.t2i)
            model = bow.CBOW(
                vocab_size=len(train_dataset.vocab.w2i),
                embedding_dim=embedding_dim,
                n_classes=n_classes,
                vocab=train_dataset.vocab,
            )
            model = model.to(device)
            print(model)
            utils.print_parameters(model)
            optimizer = optim.Adam(
                model.parameters(),
                lr=0.0001,
                weight_decay=1e-5,
                eps=1e-08,
                amsgrad=False,
            )
            prep_fn = utils.prepare_minibatch
        elif args.model == "DeepCBOW":
            embedding_dim = 300
            n_classes = len(train_dataset.vocab.t2i)
            hidden_layer = 100
            vocab_size = len(train_dataset.vocab.w2i)
            model = bow.Deep_CBOW(
                vocab_size,
                embedding_dim,
                hidden_layer,
                n_classes,
                vocab=train_dataset.vocab,
            )
            model = model.to(device)
            print(model)
            utils.print_parameters(model)
            optimizer = optim.Adam(
                model.parameters(),
                lr=0.0001,
                weight_decay=1e-5,
                eps=1e-08,
                amsgrad=False,
            )
            prep_fn = utils.prepare_minibatch
        elif args.model == "PTDeepCBOW":
            hidden_layer = 100
            n_classes = len(train_dataset.vocab.t2i)
            vocab, pretrained_vectors, embedding_dim = (
                utils.create_vocabulary_and_embeddings(args.word_embeddings)
            )
            model = bow.PTDeepCBOW(
                vocab_size=len(vocab.w2i),
                embedding_dim=embedding_dim,
                hidden_dim=hidden_layer,
                output_dim=n_classes,
                vocab=vocab,
                pretrained_vectors=pretrained_vectors,
                trainable_embeddings=args.trainable_embeddings,
            )
            model = model.to(device)
            print(model)
            utils.print_parameters(model)
            optimizer = optim.Adam(
                model.parameters(),
                lr=0.0001,
                weight_decay=1e-5,
                eps=1e-08,
                amsgrad=False,
            )
            prep_fn = utils.prepare_minibatch
        elif args.model == "LSTM":
            vocab, pretrained_vectors, embedding_dim = (
                utils.create_vocabulary_and_embeddings(args.word_embeddings)
            )
            model = lstm.LSTMClassifier(
                len(vocab.w2i), embedding_dim, 168, len(vocab.t2i), vocab
            )

            # copy pre-trained word vectors into embeddings table
            with torch.no_grad():
                model.embed.weight.data.copy_(torch.from_numpy(pretrained_vectors))
                model.embed.weight.requires_grad = args.trainable_embeddings

            print(model)
            utils.print_parameters(model)

            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=3e-4)
            prep_fn = utils.prepare_minibatch
        elif args.model == "TreeLSTM":
            vocab, pretrained_vectors, embedding_dim = (
                utils.create_vocabulary_and_embeddings(args.word_embeddings)
            )
            model = lstm.TreeLSTMClassifier(
                len(vocab.w2i), embedding_dim, 168, len(vocab.t2i), vocab
            )

            # copy pre-trained word vectors into embeddings table
            with torch.no_grad():
                model.embed.weight.data.copy_(torch.from_numpy(pretrained_vectors))
                model.embed.weight.requires_grad = args.trainable_embeddings

            print(model)
            utils.print_parameters(model)

            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=3e-4)
            prep_fn = utils.prepare_treelstm_minibatch
        else:
            raise ValueError(f"Model {args.model} not supported")

        train_epoch_losses, val_epoch_metrics, test_metrics, max_epoch = (
            utils.train_model(
                model,
                optimizer,
                train_dataset,
                dev_dataset,
                test_dataset,
                device=device,
                max_epochs=args.max_epochs,
                eval_every=args.eval_every,
                batch_size=args.batch_size,
                patience=args.patience,
                batch_fn=utils.get_minibatch,
                prep_fn=prep_fn,
                eval_fn=utils.evaluate_metrics_extended_batch,
            )
        )

        train_epoch_losses_list.append(train_epoch_losses)
        val_epoch_metrics_list.append(val_epoch_metrics)
        test_metrics_list.append(test_metrics)
        max_epochs.append(max_epoch)

    # Calculate averages and standard deviations for all metrics
    metrics_summary = {}

    # Get all metric names from the last test metrics
    metric_names = test_metrics_list[0].keys()

    for metric in metric_names:
        values = [test_metrics[metric] for test_metrics in test_metrics_list]
        metrics_summary[f"test_{metric}_mean"] = np.mean(values)
        metrics_summary[f"test_{metric}_std"] = np.std(values)

    # Print results
    print(f"Args: {args}")
    print(f"Max epochs: {max_epochs}")
    print(f"Last train loss: {round(train_epoch_losses_list[-1][-1], 2)}")
    print(f"Last val metrics: {val_epoch_metrics_list[-1][-1]}")

    # Save detailed results for the current run
    run_results = {
        "args": vars(args),  # Convert argparse.Namespace to a dictionary
        "metrics_summary": {metric_name: round(value, 5) for metric_name, value in metrics_summary.items()},
        "max_epochs": max_epochs,  # List of max epochs for each seed
        "train_epoch_losses_list": train_epoch_losses_list,  # Losses for all epochs
        "val_epoch_metrics_list": val_epoch_metrics_list  # Validation metrics for all epochs
    }

    # File naming based on arguments
    result_file = os.path.join(
        results_dir,
        f"{args.model}_{args.word_embeddings}_{args.trainable_embeddings}_{args.supervise_nodes}.json"
    )

    # Save as JSON
    with open(result_file, "w") as f:
        json.dump(run_results, f, indent=4)

    print(f"Results saved to {result_file}")

    return metrics_summary


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to train",
        choices=["LSTM", "TreeLSTM", "BOW", "CBOW", "DeepCBOW", "PTDeepCBOW"],
    )
    parser.add_argument(
        "--supervise_nodes",
        action="store_true",
        help="Use node level supervision / convert each subtree into its own sample",
        default=False,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        help="Batch size",
        default=128,
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        required=False,
        help="Maximum number of epochs",
        default=100,
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        required=False,
        help="Evaluation every n epochs",
        default=1,
    )
    parser.add_argument(
        "--patience",
        type=int,
        required=False,
        help="Patience for early stopping",
        default=10,
    )

    # word embeddings
    parser.add_argument(
        "--trainable_embeddings",
        action="store_true",
        help="Trainable embeddings",
        default=False,
    )
    parser.add_argument(
        "--word_embeddings",
        type=str,
        required=False,
        help="Word embeddings to use",
        choices=["glove", "word2vec"],
        default="glove",
    )

    args = parser.parse_args()

    main(args)
