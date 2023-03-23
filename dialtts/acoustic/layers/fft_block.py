
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_block import LinearGLU, Conv1dGLU


class MultiHeadScaleDotProductAttention(nn.Module):
    def __init__(
        self,
        num_head,
        input_size,
        kvq_size,
        dropout_rate=0.1,
    ):
        """Multi-Head Attention layer.
        Args:
            num_head: the number of head.
            input_size: the channel of input feature.
            kvq_size: the channel of each query/key/value head.
        """
        super(MultiHeadScaleDotProductAttention, self).__init__()
        self.num_head = num_head
        self.input_size = input_size
        self.kvq_size = kvq_size
        
        self.queries = LinearGLU(input_size, num_head*kvq_size)
        self.keys = LinearGLU(input_size, num_head*kvq_size)
        self.values = nn.Linear(input_size, num_head*kvq_size)
        
        self.fc = nn.Linear(num_head*kvq_size, input_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(input_size)

    def forward(self, input, masked_fill=None):
        """Compute Scaled Dot Product Attention.
        Args:
            input: (batch, time, input_size)
            masked_fill: (batch, 1, time, time)
        Returns:
            output: attentined and transformed `value` (batch, time, input_size)
                weighted by the query dot key attention (batch, head, time, time)
            attention: attention information. (batch, head, time, time)
        """
        batch = input.size(0)
        num_head, kvq_size = self.num_head, self.kvq_size
        
        q = self.queries(input).view(batch, -1, num_head, kvq_size).transpose(1, 2) # (batch, head, time, kvq_size)
        k = self.keys(input).view(batch, -1, num_head, kvq_size).transpose(1, 2) # (batch, head, time, kvq_size)
        v = self.values(input).view(batch, -1, num_head, kvq_size).transpose(1, 2) # (batch, head, time, kvq_size)
        
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(kvq_size)  # (batch, head, time, time)
        if masked_fill is not None:
            scores = scores.masked_fill(masked_fill, -np.inf)
            attention = torch.softmax(scores, dim=-1).masked_fill(masked_fill, 0.) # (batch, head, time, time)
        else:
            attention = torch.softmax(scores, dim=-1) # (batch, head, time, time)
        
        v = torch.matmul(attention, v)  # (batch, head, time, kvq_size)
        v = v.transpose(1, 2).contiguous().view(batch, -1, num_head*kvq_size)
        context = self.dropout(self.fc(v))
        output = self.norm(context + input)
        
        return output, attention


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        group_size=1,
        dropout_rate=0.1
    ):
        super(PositionwiseFeedForward, self).__init__()
        assert kernel_size % 2 == 1
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size, groups=group_size, padding=kernel_size//2)
        self.fc = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(input_size)

    def forward(self, x_, masked_fill=None):
        # x_: (B, T, C)
        # masked_fill: (B, T, 1)
        if masked_fill is not None:
            x_ = x_.masked_fill(masked_fill, 0.)
        
        x = self.conv(x_.transpose(1,2))
        x = F.relu(self.dropout(x))
        x = self.fc(x.transpose(1,2))
        x = self.norm(x + x_)
        
        if masked_fill is not None:
            x = x.masked_fill(masked_fill, 0.)
        
        return x # (B, T, C)


class FFTBlock(nn.Module):
    """FFT Block"""

    def __init__(
        self,
        num_head,
        input_size,
        kvq_size,
        filter_size,
        kernel_size,
        group_size,
        dropout_rate=0.1,
        attention_type = "scale",
    ):
        super(FFTBlock, self).__init__()
        self.num_head = num_head
        self.input_size = input_size
        self.kvq_size = kvq_size
        self.filter_size = filter_size
        
        if attention_type == "scale":
            self.attention = MultiHeadScaleDotProductAttention(num_head, input_size, kvq_size, dropout_rate)
        else:
            assert 0, f"Not support attention_type={attention_type} now!"
        self.feedforward = PositionwiseFeedForward(input_size, filter_size, kernel_size, group_size, dropout_rate)

    def forward(self, input, attn_masked_fill=None, feed_masked_fill=None):
        # input: (B, T, C)
        # attn_masked_fill: (B, 1, T, T)
        # feed_masked_fill: (B, T, 1)
        output, _ = self.attention(input, attn_masked_fill)
        output = self.feedforward(output, feed_masked_fill) # (B, T, C)
        return output


