import torch
import torch.nn as nn


class BOW(nn.Module):
    """A simple bag-of-words model"""

    def __init__(self, vocab_size, embedding_dim, vocab):
        super(BOW, self).__init__()
        self.vocab = vocab

        # this is a trainable look-up table with word embeddings
        self.embed = nn.Embedding(vocab_size, embedding_dim)

        # this is a trainable bias term
        self.bias = nn.Parameter(torch.zeros(embedding_dim), requires_grad=True)

    def forward(self, inputs):
        # this is the forward pass of the neural network
        # it applies a function to the input and returns the output

        # this looks up the embeddings for each word ID in inputs
        # the result is a sequence of word embeddings
        embeds = self.embed(inputs)

        # the output is the sum across the time dimension (1)
        # with the bias term added
        logits = embeds.sum(1) + self.bias

        return logits


class CBOW(nn.Module):
    """A continuous bag-of-words model"""

    def __init__(self, vocab_size, embedding_dim, n_classes, vocab):
        super(CBOW, self).__init__()
        self.vocab = vocab

        # Embedding layer
        self.embed = nn.Embedding(vocab_size, embedding_dim)

        # Linear layer
        self.fc = nn.Linear(embedding_dim, n_classes)

    def forward(self, inputs):
        # Get embeddings
        embeds = self.embed(inputs)  # (batch_size, sequence_length, embedding_dim)
        # Sum over sequence length
        embeds_sum = embeds.sum(dim=1)  # (batch_size, embedding_dim)
        # Linear projection
        logits = self.fc(embeds_sum)  # (batch_size, n_classes)
        return logits


class Deep_CBOW(nn.Module):
    """A deep continuous bag-of-words model"""

    def __init__(self, vocab_size, embedding_dim, hidden_layer, n_classes, vocab):
        super(Deep_CBOW, self).__init__()
        self.vocab = vocab

        # Embedding layer
        self.embed = nn.Embedding(vocab_size, embedding_dim)

        # Deep neural network for output layer
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_layer),  # E -> D
            nn.Tanh(),
            nn.Linear(hidden_layer, hidden_layer),  # D -> D
            nn.Tanh(),
            nn.Linear(hidden_layer, n_classes),  # D -> C
        )

    def forward(self, inputs):
        # Get embeddings
        embeds = self.embed(inputs)  # (batch_size, sequence_length, embedding_dim)

        # Sum over sequence length
        embeds_sum = embeds.sum(dim=1)  # (batch_size, embedding_dim)

        # Pass through the output layer
        logits = self.output_layer(embeds_sum)

        return logits


class PTDeepCBOW(Deep_CBOW):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        vocab,
        pretrained_vectors,
        trainable_embeddings=False,
    ):
        super(PTDeepCBOW, self).__init__(
            vocab_size, embedding_dim, hidden_dim, output_dim, vocab
        )

        # Load pre-trained embeddings and freeze them
        self.embed = nn.Embedding.from_pretrained(
            embeddings=torch.tensor(pretrained_vectors, dtype=torch.float32),
            freeze=not trainable_embeddings,  # Freeze embeddings to prevent updates
        )
