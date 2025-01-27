"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules import MultiHeadedStridedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout,
                 max_relative_positions=0, strided_attn=False, conv_k_v=False):
        super(TransformerEncoderLayer, self).__init__()

        self.strided_attn = strided_attn
        self.conv_k_v = conv_k_v
        if self.strided_attn:
            self.self_attn = MultiHeadedStridedAttention(
                heads, d_model, dropout=dropout,
                max_relative_positions=max_relative_positions)
        else:
            self.self_attn = MultiHeadedAttention(
                heads, d_model, dropout=dropout,
                max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.conv1d_k_v = nn.Conv1d(d_model, d_model, kernel_size=3, stride=3)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, src_len, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        q, k, v = input_norm, input_norm, input_norm
        if self.conv_k_v:
            k = self.conv1d_k_v(k.transpose(1, 2)).transpose(1, 2)
            v = self.conv1d_k_v(v.transpose(1, 2)).transpose(1, 2)
        context, _ = self.self_attn(k, v, q,
                                    mask=mask, type="self")
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings,
                 max_relative_positions, conv_first, strided_attn, conv_encoder_deconv, conv_k_v):
        super(TransformerEncoder, self).__init__()

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout,
                max_relative_positions=max_relative_positions,
                strided_attn=strided_attn, conv_k_v=conv_k_v)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.conv_first = conv_first
        self.conv_encoder_deconv = conv_encoder_deconv
        if conv_k_v or conv_encoder_deconv:
            self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, stride=3)
            self.mask_pool = nn.MaxPool1d(kernel_size=3, stride=3)
        if conv_encoder_deconv:
            self.conv_transpose = nn.ConvTranspose1d(d_model, d_model, kernel_size=3, stride=3)
            self.conv_transpose_pad1 = nn.ConvTranspose1d(d_model, d_model, kernel_size=3, stride=3, output_padding=1)
            self.conv_transpose_pad2 = nn.ConvTranspose1d(d_model, d_model, kernel_size=3, stride=3, output_padding=2)
        # assert (not(self.conv_first and self.conv_encoder_deconv))

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout,
            embeddings,
            opt.max_relative_positions,
            opt.conv_first,
            opt.strided_attn,
            opt.conv_encoder_deconv,
            opt.conv_k_v)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]

        # if set conv_first=True, convolve first for memory compressed attention and reduce seq length
        original_seq_len = out.shape[1]

        if self.conv_first or self.conv_encoder_deconv:
            out = self.conv1d(out.transpose(1, 2)).transpose(1, 2)
            mask = self.mask_pool(mask.float()).byte()

        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            out = layer(out, mask)
        out = self.layer_norm(out)

        if self.conv_encoder_deconv:
            out = out.transpose(1, 2)

            if original_seq_len % 3 == 0:
                out = self.conv_transpose(out)
            elif original_seq_len % 3 == 1:
                out = self.conv_transpose_pad1(out)
            else:
                out = self.conv_transpose_pad2(out)

            out = out.transpose(1, 2)

        return emb, out.transpose(0, 1).contiguous(), lengths
