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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main(args):

    seeds = [1, 42, 1337]

    # load data
    train_dataset = SentimentDataset(split="train", lower=False)
    dev_dataset = SentimentDataset(split="dev", lower=False)
    test_dataset = SentimentDataset(split="test", lower=False)

    train_epoch_losses_list = []
    val_epoch_accuracies_list = []
    test_accuracies = []
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
        elif args.model == "PTDeepCBOW":
            hidden_layer = 100
            n_classes = len(train_dataset.vocab.t2i)
            vocab, pretrained_vectors, embedding_dim = utils.create_vocabulary_and_embeddings(
                args.word_embeddings
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
        elif args.model == "LSTM":
            raise NotImplementedError("LSTM not implemented")
        elif args.model == "TreeLSTM":
            raise NotImplementedError("TreeLSTM not implemented")
        else:
            raise ValueError(f"Model {args.model} not supported")

        train_epoch_losses, val_epoch_accuracies, test_accuracy, max_epoch = (
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
                prep_fn=utils.prepare_minibatch,
                eval_fn=utils.evaluate_metrics_extended_batch,
            )
        )
        train_epoch_losses_list.append(train_epoch_losses)
        val_epoch_accuracies_list.append(val_epoch_accuracies)
        test_accuracies.append(test_accuracy)
        max_epochs.append(max_epoch)

    # print max_epochs, test_accuracies, average test accuracy, and standard deviation, last train accuracy and last val accuracy, rounded to 2 decimal places
    print(f"Max epochs: {max_epochs}")
    print(f"Last train accuracy: {round(train_epoch_losses_list[-1][-1], 2)}")
    print(f"Last val accuracy: {round(val_epoch_accuracies_list[-1][-1], 2)}")
    print(f"Average test accuracy: {round(np.mean(test_accuracies), 2)}")
    print(f"Standard deviation: {round(np.std(test_accuracies), 2)}")
    print(f"Test accuracies: {test_accuracies}")


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
        "--node_level",
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
