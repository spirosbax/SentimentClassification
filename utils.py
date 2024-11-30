import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm
from torch import nn
import random
import copy
import data
import requests
import os


# Here we print each parameter name, shape, and if it is trainable.
def print_parameters(model):
    total = 0
    for name, p in model.named_parameters():
        total += np.prod(p.shape)
        print(
            "{:24s} {:12s} requires_grad={}".format(
                name, str(list(p.shape)), p.requires_grad
            )
        )
    print("\nTotal number of parameters: {}\n".format(total))


def train_model(
    model,
    optimizer,
    train_data,
    dev_data,
    test_data,
    device,
    max_epochs=100,
    eval_every=1,
    batch_fn=None,
    prep_fn=None,
    eval_fn=None,
    batch_size=None,
    patience=5,
):

    criterion = nn.CrossEntropyLoss()
    train_epoch_losses = []
    val_epoch_metrics = []

    # early stopping
    best_val_accuracy = float("-inf")
    best_epoch = 0
    patience_counter = 0
    best_model = None
    try:
        for epoch in range(max_epochs):
            train_loss = 0
            num_batches = 0
            for batch in tqdm(
                batch_fn(train_data.data, batch_size=batch_size, shuffle=True)
            ):
                num_batches += 1

                # Forward pass
                model.train()
                x, targets = prep_fn(batch, model.vocab, device=device)

                logits = model(x)

                B = targets.size(0)  # Batch size

                # Compute cross-entropy loss
                loss = criterion(logits.view([B, -1]), targets.view(-1))
                train_loss += loss.item()

                # Backward pass and optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_epoch_losses.append(train_loss / num_batches)

            # validation loop
            if epoch % eval_every == 0:
                with torch.no_grad():
                    model.eval()
                    val_metrics = eval_fn(
                        model,
                        dev_data.data,
                        device=device,
                        criterion=criterion,
                        prep_fn=prep_fn,
                        batch_size=batch_size,
                        batch_fn=batch_fn,
                    )
                    val_epoch_metrics.append(val_metrics)

                print(
                    f"Epoch {epoch}, train loss: {train_epoch_losses[-1]}, "
                    f"val accuracy: {val_metrics['accuracy']}"
                )

                if val_metrics['accuracy'] > best_val_accuracy:
                    best_val_accuracy = val_metrics['accuracy']
                    best_epoch = epoch
                    patience_counter = 0
                    best_model = copy.deepcopy(model)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

        # Evaluate on test set using best model
        with torch.no_grad():
            best_model.eval()
            test_metrics = eval_fn(
                best_model,
                test_data.data,
                device=device,
                criterion=criterion,
                prep_fn=prep_fn,
                batch_size=batch_size,
                batch_fn=batch_fn,
            )
            print(f"Test metrics: {test_metrics}")

    except KeyboardInterrupt:
        print("Interrupted")

    return train_epoch_losses, val_epoch_metrics, test_metrics, epoch


def evaluate_metrics_extended_batch(
    model, data, criterion, device, batch_fn=None, prep_fn=None, batch_size=16
):
    """
    Evaluates a model on the given dataset and returns a dictionary of metrics.
    """
    model.eval()

    correct = 0
    total = 0
    y_true = []
    y_pred = []
    loss = 0
    num_batches = 0
    with torch.no_grad():  # Disable gradient computation
        for mb in batch_fn(data, batch_size=batch_size, shuffle=False):
            num_batches += 1
            # Preprocess the mini-batch to get input tensors and target labels
            x, targets = prep_fn(mb, model.vocab, device=device)

            # Forward pass: compute logits
            logits = model(x)
            B = targets.size(0)
            loss += criterion(logits.view([B, -1]), targets.view(-1)).item()

            # Get the predicted classes (as integers)
            predictions = logits.argmax(dim=-1).view(-1)

            # Update counters for accuracy
            correct += (predictions == targets.view(-1)).sum().item()
            total += targets.size(0)

            # Append true and predicted labels for F1 score computation
            y_true.extend(targets.view(-1).tolist())
            y_pred.extend(predictions.tolist())

    loss = loss / num_batches

    # Calculate all metrics
    accuracy = correct / total if total > 0 else 0.0
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    # Return metrics as a dictionary
    metrics = {
        "loss": loss / num_batches,
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
    }

    return metrics


def get_minibatch(data, batch_size=25, shuffle=True):
    """Return minibatches, optional shuffling"""

    if shuffle:
        print("Shuffling training data")
        random.shuffle(data)  # shuffle training data each epoch

    batch = []

    # yield minibatches
    for example in data:
        batch.append(example)

        if len(batch) == batch_size:
            yield batch
            batch = []

    # in case there is something left
    if len(batch) > 0:
        yield batch


def prepare_minibatch(mb, vocab, device):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """
    batch_size = len(mb)
    maxlen = max([len(ex.tokens) for ex in mb])

    # vocab returns 0 if the word is not there
    x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen) for ex in mb]

    x = torch.LongTensor(x)
    x = x.to(device)

    y = [ex.label for ex in mb]
    y = torch.LongTensor(y)
    y = y.to(device)

    return x, y

def prepare_treelstm_minibatch(mb, vocab, device):
  """
  Returns sentences reversed (last word first)
  Returns transitions together with the sentences.
  """
  batch_size = len(mb)
  maxlen = max([len(ex.tokens) for ex in mb])

  # vocab returns 0 if the word is not there
  # NOTE: reversed sequence!
  x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen)[::-1] for ex in mb]

  x = torch.LongTensor(x)
  x = x.to(device)

  y = [ex.label for ex in mb]
  y = torch.LongTensor(y)
  y = y.to(device)

  maxlen_t = max([len(ex.transitions) for ex in mb])
  transitions = [pad(ex.transitions, maxlen_t, pad_value=2) for ex in mb]
  transitions = np.array(transitions)
  transitions = transitions.T  # time-major

  return (x, transitions), y

def pad(tokens, length, pad_value=1):
    """add padding 1s to a sequence to that it has the desired length"""
    return tokens + [pad_value] * (length - len(tokens))



def batch(states):
    """
    Turns a list of states into a single tensor for fast processing.
    This function also chunks (splits) each state into a (h, c) pair"""
    return torch.cat(states, 0).chunk(2, 1)

def unbatch(state):
    """
    Turns a tensor back into a list of states.
    First, (h, c) are merged into a single state.
    Then the result is split into a list of sentences.
    """
    return torch.split(torch.cat(state, 1), 1, 0)


def download_file(url, output_file):
    """
    Downloads a file from a given URL and saves it to the specified output file.
    """
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors

    with open(output_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):  # Stream content in chunks
            file.write(chunk)
    print(f"File saved as {output_file}")


def load_pretrained_embeddings(file_path):
    """
    Loads pre-trained word embeddings from a file.
    Args:
        file_path: Path to the embedding file.
    Returns:
        embedding_dict: Dictionary mapping words to their embeddings.
        embedding_dim: The dimension of the embeddings.
    """

    # URLs for the filtered versions of GloVe and Word2Vec embeddings
    glove_url = "https://gist.githubusercontent.com/bastings/b094de2813da58056a05e8e7950d4ad1/raw/3fbd3976199c2b88de2ae62afc0ecc6f15e6f7ce/glove.840B.300d.sst.txt"
    word2vec_url = "https://gist.githubusercontent.com/bastings/4d1c346c68969b95f2c34cfbc00ba0a0/raw/76b4fefc9ef635a79d0d8002522543bc53ca2683/googlenews.word2vec.300d.txt"

    # Output file names
    glove_file = "glove.840B.300d.sst.txt"
    word2vec_file = "googlenews.word2vec.300d.txt"

    # Download files
    # Download the file if it doesn't exist
    if not os.path.exists(file_path):
        download_file(glove_url, glove_file)
        download_file(word2vec_url, word2vec_file)

    embedding_dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            split_line = line.strip().split()
            if len(split_line) == 0:
                continue  # Skip empty lines
            word = split_line[0]
            vector = np.array(split_line[1:], dtype=np.float32)
            embedding_dict[word] = vector
    embedding_dim = len(next(iter(embedding_dict.values())))  # Get the embedding size
    return embedding_dict, embedding_dim

def create_vocabulary_and_embeddings(embeddings, verbose=True):
    """
    Creates a vocabulary and embedding matrix from pretrained embeddings.

    Args:
        embeddings: string, either "glove" or "word2vec"
        verbose: Whether to print summary statistics

    Returns:
        vocab: The created Vocabulary object
        vectors: numpy array of embeddings corresponding to the vocabulary
        embedding_dim: The dimension of the embeddings
    """
    # Initialize a new Vocabulary object
    vocab = data.Vocabulary()

    # Load the embeddings
    glove_file = "glove.840B.300d.sst.txt"
    word2vec_file = "googlenews.word2vec.300d.txt"
    embeddings_file = {"glove": glove_file, "word2vec": word2vec_file}
    embedding_dict, embedding_dim = load_pretrained_embeddings(
        embeddings_file[embeddings]
    )

    # Initialize <unk> and <pad> embeddings
    unk_embedding = np.random.uniform(-0.1, 0.1, embedding_dim).astype(np.float32)
    pad_embedding = np.zeros(embedding_dim, dtype=np.float32)

    # Add the corresponding embeddings to a list
    vectors = []

    # Add <unk> and <pad> tokens to the vocabulary and their embeddings
    vocab.add_token("<unk>")  # Index 0
    vectors.append(unk_embedding)
    vocab.add_token("<pad>")  # Index 1
    vectors.append(pad_embedding)

    # Populate the vocabulary and embedding matrix
    for word, vector in embedding_dict.items():
        vocab.add_token(word)
        vectors.append(vector)

    # Convert vectors to a numpy matrix
    vectors = np.stack(vectors, axis=0)

    if verbose:
        print("Vocabulary size: ", len(vocab.w2i))
        print(f"Embedding matrix shape: {vectors.shape}")

    return vocab, vectors, embedding_dim
