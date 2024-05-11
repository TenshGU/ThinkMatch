import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        """
        MultiHeadAttention mechanism. The input of the MultiHeadAttention mechanism is an embedding (or sequence of embeddings).
        The embeddings are split into different parts and each part is fed into a head.
        :param embed_size: the size of the embedding.
        :param heads: the number of heads you wish to create.
        """
        self.embed_size = embed_size  # 512 in Transformer
        self.heads = heads  # 8 in Transformer
        self.head_dim = embed_size // heads  # 64 in Transformer
        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        # === Project Embeddings into three vectors: Query, Key and Value ===
        # Note: some implementations do: nn.Linear(embed_size, head_dim). We won't do this. We will project it
        # on a space of size embed_size and then split it into N heads of head_dim shape.
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        # Values, Keys: target graph points
        # Queries: source graph points
        # Values, Keys and Queries have size: (batch_size, pointset_size, embedding_size)
        batch_size = query.shape[0]  # Get number of training examples/batch size.
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # === Pass through Linear Layer ===
        values = self.values(values)  # (batch_size, value_len, embed_size)
        keys = self.keys(keys)  # (batch_size, key_len, embed_size)
        queries = self.queries(query)  # (batch_size, query_len, embed_size)

        # Split the embedding into self.heads different pieces
        values = values.reshape(batch_size, value_len, self.heads, self.head_dim)
        keys = keys.reshape(batch_size, key_len, self.heads, self.head_dim)
        queries = queries.reshape(batch_size, query_len, self.heads, self.head_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (batch_size, query_len, heads, heads_dim),
        # keys shape: (batch_size, key_len, heads, heads_dim)
        # energy: (batch_size, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (batch_size, heads, query_len, key_len)
        # values shape: (batch_size, value_len, heads, heads_dim)
        # out after matrix multiply: (batch_size, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            batch_size, query_len, self.heads * self.head_dim
        )
        # Linear layer doesn't modify the shape, final shape will be
        # (batch_size, query_len, embed_size)
        out = self.fc_out(out)
        return out


class TransformerLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion=4):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # Values, Keys and Queries have size: (batch_size, query_len, embedding_size)
        attention = self.attention(value, key, query, mask)  # attention shape: (batch_size, query_len, embedding_size)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))  # x shape: (batch_size, query_len, embedding_size)
        forward = self.feed_forward(x)  # forward shape: (batch_size, query_len, embedding_size)
        out = self.dropout(self.norm2(forward + x))  # out shape: (batch_size, query_len, embedding_size)
        return out


class Encoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout):
        super(Encoder, self).__init__()
        self.embed_size = embed_size  # size of the input embedding
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass.
        :param x: source sequence. Shape: (batch_size, source_sequence_len).
        :return output: torch tensor of shape (batch_size, src_sequence_length, embedding_size)
        """
        out = self.dropout(x)
        # In the Encoder the query, key, value are all the same, in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out, None)
        # output shape: torch.Size([batch_size, sequence_length, embedding_size])
        return out


class DecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderLayer, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerLayer(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, value, key, src_mask):
        """
        :param query: target input. Shape: (batch_size, target_graph_pointset_size, embedding_size)
        :param value: value extracted from encoder.
        :param key: key extracted from encoder.
        :param src_mask: source mask is used when we need to pad the input.
        """
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, pointset_avg_size, forward_expansion, dropout):
        """
        :param embed_size: size of output embeddings.
        :param num_layers: number of DecoderLayers in the Decoder.
        :param heads: number of heads in the MultiAttentionHeads inside the DecoderLayer.
        :param pointset_avg_size: the average size of pointset.
        :param forward_expansion: expansion factor in LinearLayer at the end of the TransformerLayer.
        :param dropout: dropout probability.
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(embed_size, heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, pointset_avg_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, trg_mask):
        """
        :param x: target sequence. Shape: (batch_size, target_graph_pointset_size, embedding_size)
        :param enc_out: encoder output. Shape: (batch_size, src_pointset_size, embedding_size)
        :param trg_mask: target mask.
        """
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, trg_mask)

        out = self.fc_out(x)
        return out


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()

        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ARNE(nn.Module):
    def __init__(self, embed_size=128, pointset_avg_size=30,
                 num_layers=3, forward_expansion=4, heads=4, dropout=0, device="cpu"):
        super(ARNE, self).__init__()
        # === Encoder ===
        self.encoder = Encoder(embed_size, num_layers, heads, forward_expansion, dropout)
        # === Decoder ===
        self.decoder = Decoder(embed_size, num_layers, heads, pointset_avg_size, forward_expansion, dropout)
        self.MLP = MLP(embed_size, embed_size / 2, pointset_avg_size)
        self.device = device

    def make_trg_mask(self, trg):
        """
        Returns a lower triangular matrix filled with 1s. The shape of the mask is (target_size, target_size).
        Example: for a target of shape (batch_size=1, target_size=4)
        tensor([[[[1., 0., 0., 0.],
                  [1., 1., 0., 0.],
                  [1., 1., 1., 0.],
                  [1., 1., 1., 1.]]]])
        """
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        trg_mask = self.make_trg_mask(trg)  # trg_mask shape:
        enc_src = self.encoder(src)  # enc_src shape:
        out = self.decoder(trg, enc_src, trg_mask)  # out shape:
        out = self.MLP(out)
        return out

