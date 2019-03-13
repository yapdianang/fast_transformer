import math
import torch
import torch.nn as nn

from onmt.utils.misc import generate_relative_positions_matrix,\
                            relative_matmul

class MultiHeadedStridedAttention(nn.Module):

    def __init__(self, head_count, model_dim, dropout=0.1,
                 max_relative_positions=0):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedStridedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.max_relative_positions = max_relative_positions

        if max_relative_positions > 0:
            vocab_size = max_relative_positions * 2 + 1
            self.relative_positions_embeddings = nn.Embedding(
                vocab_size, self.dim_per_head)

    def perform_attention(self, 
                          query, dim_per_head, 
                          key, relations_keys, mask, 
                          value, relations_values, 
                          batch_size, head_count, 
                          query_len, key_len,
                          shape, unshape):

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        # batch x num_heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))

        if self.max_relative_positions > 0 and type == "self":
            scores = query_key + relative_matmul(query, relations_keys, True)
        else:
            scores = query_key
        scores = scores.float()

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)

        if self.max_relative_positions > 0 and type == "self":
            context = unshape(context_original
                              + relative_matmul(drop_attn,
                                                relations_values,
                                                False))
        else:
            context = unshape(context_original)

        output = self.final_linear(context)

        # Return one attn
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()

        return output, top_attn

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None):

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)
        device = key.device

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)
                key = shape(key)
                value = shape(value)
                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"].to(device), key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"].to(device), value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["memory_keys"],\
                               layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        relations_keys = None
        relations_values = None
        if self.max_relative_positions > 0 and type == "self":
            key_len = key.size(2)
            # 1 or key_len x key_len
            relative_positions_matrix = generate_relative_positions_matrix(
                key_len, self.max_relative_positions,
                cache=True if layer_cache is not None else False)
            #  1 or key_len x key_len x dim_per_head
            relations_keys = self.relative_positions_embeddings(
                relative_positions_matrix.to(device))
            #  1 or key_len x key_len x dim_per_head
            relations_values = self.relative_positions_embeddings(
                relative_positions_matrix.to(device))

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)


        splice_inds = [[0, query_len//2], [query_len//4, ((query_len//4)*3)], [query_len//2, query_len]]

        outputs = []
        top_attns = []
        for splice in splice_inds:
          q_len = splice[1] - splice[0]
          query_split = query[:, :, splice[0]:splice[1], :]
          key_split = key[:, :, splice[0]:splice[1], :]
          value_split = value[:, :, splice[0]:splice[1], :]
          mask_split = mask[:, :, splice[0]:splice[1]]
          output, top_attn = self.perform_attention(query_split, dim_per_head, key_split, relations_keys, mask_split, value_split, relations_values, batch_size, head_count, q_len, q_len, shape, unshape)
          outputs.append(output)
          top_attns.append(top_attn)

        # TODO: add .cuda() after to run on GPU
        output = torch.zeros((query.shape[0], query.shape[2], outputs[0].shape[2]))
        output[:, splice_inds[1][0]:splice_inds[1][1], :] = outputs[1]
        output[:, 0:(outputs[0].shape[1]//3)*2, :] = outputs[0][:, 0:(outputs[0].shape[1]//3)*2, :]

        amt = (outputs[2].shape[1]//3)*2
        output[:, -amt:, :] = outputs[2][:, -amt:, :]

        print(output.shape)
        print(output[0, :, 0])
        raise "TEST"

        return output, top_attn

        # NOTE: Relations_keys and relations_values shapes should be None
        # output, top_attn = self.perform_attention(query, dim_per_head, 
                          # key, relations_keys, mask, 
                          # value, relations_values, 
                          # batch_size, head_count, 
                          # query_len, key_len,
                          # shape, unshape)
        # return output, top_attn
