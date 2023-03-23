# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dialtts.acoustic.layers import LinearGLU, Conv1dGLU
from dialtts.acoustic.layers import FFTBlock, GCLBlock
from dialtts.acoustic.layers import MaskedGroupNorm, ChannelNorm
from dialtts.acoustic.utils import get_mask_from_lengths


class FFTNet(nn.Module):
    def __init__(
        self,
        num_layer=4,
        num_head=4,
        num_hidden=256,
        filter_size=512,
        kernel_size=9,
        group_size=4,
        dropout_rate=0.1,
        attention_type = "scale",
    ):
        super(FFTNet, self).__init__()
        self.num_head = num_head
        
        self.blocks = nn.ModuleList([
            FFTBlock(
                num_head, num_hidden, num_hidden//num_head,
                filter_size, kernel_size, group_size,
                dropout_rate=dropout_rate,
                attention_type=attention_type,
            ) for _ in range(num_layer)
        ])
    
    def forward(self, inputs, mask=None):
        # inputs: (B, N, c)
        # mask: (B, N, 1)
        if mask is not None:
            attn_masked_fill = mask.repeat(1, 1, inputs.size(1)) # (B, N, N)
            attn_masked_fill = attn_masked_fill.unsqueeze(1) # (B, 1, N, N)
            attn_masked_fill = ~attn_masked_fill
            feed_masked_fill = ~mask # (B, N, 1)
        else:
            attn_masked_fill, feed_masked_fill = None, None
        
        encoded = inputs
        for f in self.blocks:
            encoded = f(encoded, attn_masked_fill, feed_masked_fill)
        
        return encoded # (B, N, c)


class GCLNet(nn.Module):
    def __init__(
        self,
        num_input, *,
        num_layer=4,
        num_hidden=256,
        kernel_size=17,
        group_size=4,
        dropout_rate=0.1,
    ):
        super(GCLNet, self).__init__()
        
        self.blocks = nn.ModuleList([
            GCLBlock(
                num_input, num_hidden,
                kernel_size=kernel_size,
                groups=group_size,
                dropout_rate=dropout_rate
            ) for _ in range(num_layer)
        ])
        
    def forward(self, x, mask=None):
        # x: (B, C, T)
        # mask: (B, T, 1)
        if mask is not None:
            masked_fill = ~mask.transpose(1,2)
            x = x.masked_fill(masked_fill, 0.)
        else:
            masked_fill = None
        
        for f in self.blocks:
            x = f(x, masked_fill)
        
        return x # (B, C, T)


class TextEncoder(nn.Module):
    def __init__(
        self,
        num_text, *,
        num_layer=4,
        num_head=4,
        num_hidden=256,
        filter_size=512,
        kernel_size=9,
        group_size=4,
        dropout_rate=0.1,
        attention_type="scale",
    ):
        super(TextEncoder, self).__init__()
        self.fc = nn.Sequential(
            LinearGLU(num_text, num_hidden),
            nn.LayerNorm(num_hidden),
        )
        self.fftnet = FFTNet(
            num_layer=num_layer,
            num_head=num_head,
            num_hidden=num_hidden,
            filter_size=filter_size,
            kernel_size=kernel_size,
            group_size=group_size,
            dropout_rate=dropout_rate,
            attention_type=attention_type,
        )
    
    def forward(self, x, mask=None):
        # x: (B, N, C)
        # mask: (B, N, 1)
        x = self.fc(x)
        x = self.fftnet(x, mask)
        return x # (B, N, C)


class MelDecoder(nn.Module):
    def __init__(
        self,
        num_mel, *,
        num_layer=4,
        num_head=4,
        num_hidden=256,
        filter_size=512,
        kernel_size=9,
        group_size=4,
        dropout_rate=0.1,
        attention_type="scale",
    ):
        super(MelDecoder, self).__init__()
        self.fftnet = FFTNet(
            num_layer=num_layer,
            num_head=num_head,
            num_hidden=num_hidden,
            filter_size=filter_size,
            kernel_size=kernel_size,
            group_size=group_size,
            dropout_rate=dropout_rate,
            attention_type=attention_type,
        )
        self.mpg = Conv1dGLU(num_hidden, num_hidden, 1)
        self.mel = nn.Conv1d(num_hidden, num_mel, 1)
    
    def forward(self, x, mask=None):
        # x: (B, T, C)
        # mask: (B, T, 1)
        x = self.fftnet(x, mask)
        mpg = self.mpg(x.transpose(1,2)) # (B, C, T)
        mel = self.mel(mpg) # (B, num_mel, T)
        return mel, mpg


class PostNet(nn.Module):
    def __init__(
        self,
        num_mel, *,
        num_layer=4,
        num_input=256,
        num_hidden=256,
        kernel_size=17,
        group_size=4,
        dropout_rate=0.1,
    ):
        super(PostNet, self).__init__()
        self.gclnet = GCLNet(
            num_layer=num_layer,
            num_input=num_input,
            num_hidden=num_hidden,
            kernel_size=kernel_size,
            group_size=group_size,
            dropout_rate=dropout_rate,
        )
        self.mpg = Conv1dGLU(num_hidden, num_hidden, 1)
        self.mel = nn.Conv1d(num_hidden, num_mel, 1)
    
    def forward(self, x, mask=None):
        # x: (B, C, T)
        # mask: (B, T, 1)
        x = self.gclnet(x, mask)
        mpg = self.mpg(x) # (B, C, T)
        mel = self.mel(mpg) # (B, num_mel, T)
        return mel, mpg


class EnergyNet(nn.Module):
    def __init__(
        self,
        num_src,
        num_trg,
        num_output,
        num_layer=4,
        num_head=4,
        num_hidden=256,
        filter_size=512,
        kernel_size=9,
        group_size=4,
        dropout_rate=0.1,
        attention_type="scale",
    ):
        super(EnergyNet, self).__init__()
        
        self.num_src = num_src
        
        self.fc0 = LinearGLU(num_src, num_src)
        self.fc1 = LinearGLU(num_trg, num_hidden)
        self.fftnet = FFTNet(
            num_layer=num_layer,
            num_head=num_head,
            num_hidden=num_hidden,
            filter_size=filter_size,
            kernel_size=kernel_size,
            group_size=group_size,
            dropout_rate=dropout_rate,
            attention_type=attention_type
        )
        self.fc2 = LinearGLU(num_hidden, num_src)
        self.fc3 = LinearGLU(num_src, num_output)
        
    def forward(self, source, target, mask_src, mask_trg):
        # source: (B,N,c)
        # target: (B,T,d)
        # mask_src: (B,N,1)
        # mask_trg: (B,T,1)
        mask = torch.bmm(mask_src.float(), mask_trg.transpose(1,2).float()).bool() # (B,N,T)
        masked_fill = ~mask
        masked_trg = ~mask_trg
        
        query = self.fc0(source) # (B,N,c)
        
        encoded = self.fc1(target).masked_fill(masked_trg, 0.)
        encoded = self.fftnet(encoded, mask_trg)
        key = self.fc2(encoded).masked_fill(masked_trg, 0.) # (B,T,c)
        value = self.fc3(key).masked_fill(masked_trg, 0.) # (B,T,e)
        
        scores = torch.matmul(query, key.transpose(1,2)) / math.sqrt(self.num_src) # (B,N,T)
        scores = scores.masked_fill(masked_fill, -np.inf)
        attention = torch.softmax(scores, dim=-1).masked_fill(masked_fill, 0.) # (B,N,T)
        context = torch.matmul(attention, value) # (B,N,e)
        
        return context, attention


class PositionalEncoding(nn.Module):
    """Postional encoding
        PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
        PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """
    def __init__(
        self,
        d_model,
        max_len=1024,
    ):
        super(PositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        self.max_len = max_len
        
        self.register_buffer(
            "sin_table",
            self._get_sin_table(max_len, d_model),
            persistent=False,
        )
    
    def _get_sin_table(self, max_len, d_model, padding_idx=None):
        """Sinusiod postional encoding table"""
        pe = torch.torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        if padding_idx is not None:
            pe[padding_idx] = 0.0
        
        return pe.unsqueeze_(0) # (1, max_len, d_model)
    
    def forward(self, x, mask=None):
        # x: (B, T, d)
        # mask: (B, T, 1)
        T = x.size(1)
        
        if T <= self.max_len:
            pe = self.sin_table[:, :T]
        elif self.training:
            assert T < self.max_len, f"The input T={T} > max_len{self.max_len}, pls resolve it!"
        else:
            pe = self._get_sin_table(T, self.d_model).to(x.device)
        
        x = x * self.xscale + pe
        if mask is not None:
            x = x.masked_fill(~mask, 0.)
            
        return x


class LearnedUpsamplingNet(nn.Module):
    def __init__(self, M=256):
        super(LearnedUpsamplingNet, self).__init__()
        
        self.conv_w = Conv1dGLU(M, 8, 3)
        self.inorm_w = MaskedGroupNorm(8, 8)
        self.conv_c = Conv1dGLU(M, 8, 3)
        self.inorm_c = MaskedGroupNorm(8, 8)
        
        self.mlp_w = nn.Sequential(
            LinearGLU(1+1+8, 16),
            LinearGLU(16, 16),
            nn.Linear(16, 1),
        )
        
        self.mlp_c = nn.Sequential(
            LinearGLU(1+1+8, 2),
            LinearGLU(2, 2),
        )
        
        self.A = nn.Linear(2, M)
    
    def forward(self, V, d, mask_K=None, mask_T=None):
        # V: (B,M,K), that is eq to (B,c,N)
        # d: (B,K)
        # mask_K: (B,1,K)
        # mask_T: (B,T,1)
        
        B, K = V.size(0), d.size(1)
        T = max(torch.sum(d.long(), dim=1))
        
        if mask_K is not None and mask_T is not None:
            assert T == mask_T.size(1) and K == mask_K.size(2)
            mask = torch.bmm(mask_T.float(), mask_K.float()).bool() # (B,T,K)
        else:
            mask = None
        
        # Token Boundaries
        e = torch.cumsum(d, dim=1) # (B,K)
        s = e - d # (B,K)
        
        # Grid matrices S and E
        S = torch.ones(B, T, K, device=V.device, dtype=torch.float)
        S = torch.cumsum(S, dim=1)
        E = S * -1.
        S = S - 1. - s.view(B,1,K)
        E = E + 1. + e.view(B,1,K)
        
        # Conv1d(V)
        V_w = self.inorm_w(self.conv_w(V), mask_K) # (B,8,K)
        V_c = self.inorm_c(self.conv_c(V), mask_K) # (B,8,K)
        V_w = V_w.transpose(1,2).unsqueeze(1) # (B,1,K,8)
        V_c = V_c.transpose(1,2).unsqueeze(1) # (B,1,K,8)
        
        # Concat(S,E,Conv1d(V))
        V_w = V_w.expand(-1,T,-1,-1) # (B,T,K,8)
        V_c = V_c.expand(-1,T,-1,-1) # (B,T,K,8)
        S, E = S.view(B,T,K,1), E.view(B,T,K,1)
        SEV_w = torch.cat((S,E,V_w), dim=-1).contiguous() # (B,T,K,10)
        SEV_c = torch.cat((S,E,V_c), dim=-1).contiguous() # (B,T,K,10)
        
        # W = Softmax(MLP(S, E, Conv1d(V)))
        W = self.mlp_w(SEV_w).squeeze(-1) # (B,T,K)
        if mask is not None:
            masked_fill = ~mask
            W = W.masked_fill(masked_fill, -np.inf)
            W = torch.softmax(W, dim=-1).masked_fill(masked_fill, 0.)
        else:
            W = torch.softmax(W, dim=-1)
        
        # C = MLP(S, E, Conv1d(V))
        C = self.mlp_c(SEV_c) # (B,T,K,2)
        
        # O = W @ V + [W*C1 ... W*Cp] @ A
        O = torch.matmul(W, V.transpose(1,2)) # (B,T,M)
        
        C0, C1 = W * C[..., 0], W * C[..., 1] # (B,T,K)
        C0 = torch.sum(C0, dim=-1, keepdim=True) # (B,T,1)
        C1 = torch.sum(C1, dim=-1, keepdim=True) # (B,T,1)
        C = torch.cat((C0,C1), dim=-1) # (B,T,2)
        O = O + self.A(C) # (B,T,M) eq to (B,T,c)
        
        if mask_T is not None:
            O = O.masked_fill(~mask_T, 0.)
        
        return O


class PhonemeAdaptor(nn.Module):
    def __init__(
        self,
        speaker_size=256, *,
        num_layer=2,
        num_head=4,
        num_hidden=256,
        filter_size=512,
        kernel_size=9,
        group_size=4,
        dropout_rate=0.1,
        attention_type="scale",
    ):
        super(PhonemeAdaptor, self).__init__()
        
        self.fc = nn.Linear(speaker_size, num_hidden)
        self.fftnet = FFTNet(
            num_layer=num_layer,
            num_head=num_head,
            num_hidden=num_hidden,
            filter_size=filter_size,
            kernel_size=kernel_size,
            group_size=group_size,
            dropout_rate=dropout_rate,
            attention_type=attention_type,
        )
        
    def forward(self, x, s, mask=None):
        # x: (B,N,c)
        # s: (B,s)
        # mask: (B,N,1)
        s = self.fc(s).unsqueeze(1) # (B,1,c)
        x = x + s # (B,N,c)
        x = self.fftnet(x, mask) # (B,N,c)
        return x


class UtteranceAdaptor(nn.Module):
    def __init__(
        self,
        speaker_size=256, *,
        num_layer=2,
        num_head=4,
        num_hidden=256,
        filter_size=512,
        kernel_size=9,
        group_size=4,
        dropout_rate=0.1,
        attention_type="scale",
    ):
        super(UtteranceAdaptor, self).__init__()
        
        self.fc = nn.Linear(speaker_size, num_hidden)
        self.fftnet = FFTNet(
            num_layer=num_layer,
            num_head=num_head,
            num_hidden=num_hidden,
            filter_size=filter_size,
            kernel_size=kernel_size,
            group_size=group_size,
            dropout_rate=dropout_rate,
            attention_type=attention_type,
        )
        
    def forward(self, x, s, mask=None):
        # x: (B,T,d)
        # s: (B,s)
        # mask: (B,T,1)
        s = self.fc(s).unsqueeze(1) # (B,1,d)
        x = x + s # (B,T,d)
        x = self.fftnet(x, mask) # (B,T,d)
        return x


class VarianceAdaptor(nn.Module):
    def __init__(
        self,
        num_hidden,
        num_energy=8,
    ):
        super(VarianceAdaptor, self).__init__()
        
        self.duration_predictor = nn.Sequential(
            Conv1dGLU(num_hidden, num_hidden, 1),
            nn.Conv1d(num_hidden, 1, 1),
        )
        
        self.energy_predictor = nn.Sequential(
            Conv1dGLU(num_hidden, num_hidden, 3),
            Conv1dGLU(num_hidden, num_energy, 1),
        )
        
        self.energy_variance = nn.Conv1d(num_energy, num_hidden, 1)
        
        self.upsample = LearnedUpsamplingNet(num_hidden)
    
    def forward(
        self,
        encoded,    # (B,N,c)
        text_mask,  # (B,N,c)
        mel_mask,   # (B,T,1)
        duration_target, # (B,N)
        energy_target, # (B,8,N)
    ):
        encoded = encoded.transpose(1,2) # (B,c,N)
        text_mask = text_mask.transpose(1,2) # (B,1,N)
        
        # duration
        duration_prediction = self.duration_predictor(encoded) # (B,1,N)
        duration_prediction = duration_prediction.masked_fill(~text_mask, 0.).squeeze(1) #(B,N)
        
        # energy
        energy_prediction = self.energy_predictor(encoded) # (B,8,N)
        variance = self.energy_variance(energy_target)
        
        # add variance
        encoded = encoded + variance # (B,c,N)
        
        # upsample
        expanded = self.upsample(encoded, duration_target.long(), text_mask, mel_mask) # expanded (B,T,c)
        
        return expanded, duration_prediction, energy_prediction
    
    def infer(self, encoded, duration_rate=1.0):
        # encoded: (B=1,N,c)
        encoded = encoded.transpose(1,2)
        
        # duration
        duration_prediction = self.duration_predictor(encoded).squeeze(1) # (B,N)
        duration_prediction = torch.exp(duration_prediction) * duration_rate
        duration_rounded = torch.round(duration_prediction)
        mask = duration_rounded < 1
        duration_rounded = duration_rounded.masked_fill(mask, 1.) # more than one frame for each phoneme
        
        # energy
        energy_prediction = self.energy_predictor(encoded) # (B,8,N)
        variance = self.energy_variance(energy_prediction)
        
        # add variance
        encoded = encoded + variance # (B,c,N)
        
        # upsample
        expanded = self.upsample(encoded, duration_rounded.long()) # expanded (B,T,c)
        
        return expanded, duration_rounded


class FasTTS(nn.Module):
    def __init__(
        self,
        num_speaker=1024,
        num_text=192,
        num_mel=48,
        text_encoder_params={
            "num_layer": 4,
            "num_head": 4,
            "num_hidden": 256,
            "filter_size": 512,
            "kernel_size": 9,
            "group_size": 4,
            "dropout_rate": 0.1,
        },
        mel_decoder_params={
            "num_layer": 4,
            "num_head": 4,
            "num_hidden": 256,
            "filter_size": 512,
            "kernel_size": 9,
            "group_size": 4,
            "dropout_rate": 0.1,
        },
        postnet_params={
            "num_layer": 4,
            "num_input": 256,
            "num_hidden": 256,
            "kernel_size": 17,
            "group_size": 4,
            "dropout_rate": 0.1,
        },
        phoneme_adaptor_params={
            "num_layer": 2,
            "num_head": 4,
            "num_hidden": 256,
            "filter_size": 512,
            "kernel_size": 9,
            "group_size": 4,
            "dropout_rate": 0.1,
        },
        utterance_adaptor_params={
            "num_layer": 2,
            "num_head": 4,
            "num_hidden": 256,
            "filter_size": 512,
            "kernel_size": 9,
            "group_size": 4,
            "dropout_rate": 0.1,
        },
        variance_adaptor_params={
            "num_energy": 8,
        },
        energy_params={
            "num_layer": 4,
            "num_head": 4,
            "num_hidden": 256,
            "filter_size": 512,
            "kernel_size": 9,
            "group_size": 4,
            "dropout_rate": 0.1,
        },
    ):
        super(FasTTS, self).__init__()
        
        hidden_size = text_encoder_params["num_hidden"]
        
        # positional encoding
        self.positional_encoding = PositionalEncoding(hidden_size, 1024)
        
        # speaker embedding
        self.speaker_embedding = nn.Embedding(num_speaker, hidden_size)
        
        # text embedding & encoder
        self.text_embedding = nn.Sequential(
            nn.Linear(num_text, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.text_encoder = TextEncoder(hidden_size, **text_encoder_params)
        
        # mel decoder
        self.mel_decoder = MelDecoder(num_mel, **mel_decoder_params)
        
        # postnet
        self.postnet = PostNet(num_mel, **postnet_params)
        
        # phoneme and utterance adaptor
        self.phoneme_adaptor = PhonemeAdaptor(hidden_size, **phoneme_adaptor_params)
        self.utterance_adaptor = UtteranceAdaptor(hidden_size, **utterance_adaptor_params)
        
        # variance adaptor
        self.variance_adaptor = VarianceAdaptor(hidden_size, **variance_adaptor_params)
        
        # energy
        self.energy = EnergyNet(hidden_size, num_mel, variance_adaptor_params["num_energy"], **energy_params)
        
    
    def forward(self, spkid, text_padded, duration_padded, mel_padded, steps=0):
        # spkid: (B,)
        # text_padding: (B,N,c)
        # duration_padded: (B,N)
        # mel_padded: (B,d,T)
        spk_embed = self.speaker_embedding(spkid) # (B,s)
        text_embed = self.text_embedding(text_padded) # (B,N,c)
        
        # duration_padded: batch of frame number of each phoneme. (B,N)
        # text_lengths: batch of phoneme number. (B,)
        # mel_lengths: batch of frame numeber. (B,)
        text_lengths = torch.sum(duration_padded>0, dim=1)
        mel_lengths = torch.sum(duration_padded, dim=1)
        N, T = torch.max(text_lengths.long()), torch.max(mel_lengths.long())
        assert N == text_embed.size(1), f"N={N}, text_embed={text_embed.size(1)}"
        
        text_mask = get_mask_from_lengths(text_lengths, N, text_embed.device).unsqueeze(2) # (B,N,1)
        mel_mask = get_mask_from_lengths(mel_lengths, T, text_embed.device).unsqueeze(2) # (B,T,1)
        
        # text encoding
        text_embed = self.positional_encoding(text_embed, text_mask)
        text_encoded = self.text_encoder(text_embed, text_mask) # (B,N,c)
        
        # phoneme adaptor
        text_encoded = self.phoneme_adaptor(text_encoded, spk_embed, text_mask) # (B,N,c)
        
        # energy: phoneme level latent
        energy_target, _ = self.energy(
            text_encoded,
            mel_padded.transpose(1,2),
            text_mask, mel_mask
        )
        energy_target = energy_target.transpose(1,2) # (B,8,N)
        
        # variance adaptor and upsampling
        text_expanded, duration_prediction, energy_prediction = self.variance_adaptor(
            text_encoded,
            text_mask, mel_mask,
            duration_target=duration_padded,
            energy_target=energy_target,
        )
        assert T == text_expanded.size(1), f"T={T}, text_expanded={text_expanded.size(1)}"
        
        # utterance adaptor
        text_expanded = self.utterance_adaptor(text_expanded, spk_embed, mel_mask) # (B,T,d)
        
        # mel decoding
        text_expanded = self.positional_encoding(text_expanded, mel_mask)
        mel_output1, mpg_output1 = self.mel_decoder(text_expanded, mel_mask) # text_expanded: (B,T,d)
        
        # postnet
        mel_output2, _ = self.postnet(mpg_output1, mel_mask)
        
        return (
            [mel_output1, mel_output2], # [(B,d,T),...]
            (duration_prediction, energy_target, energy_prediction), # (B,N)/(B,8,N)/(B,8,N)
            (text_mask.squeeze(-1), mel_mask.squeeze_(-1)), # (B,N)/(B,1,T)
        )
    
    @torch.no_grad()
    def infer(self, spkid, text, duration_rate=1.0):
        # spkid: (B=1,)
        # text: (B=1,N,c)
        spk_embed = self.speaker_embedding(spkid) # (B=1,s)
        text_embed = self.text_embedding(text) # (B=1,N,c)
        
        # text encoding
        text_embed = self.positional_encoding(text_embed)
        text_encoded = self.text_encoder(text_embed) # (B=1,N,c)
        
        # phoneme adaptor
        text_encoded = self.phoneme_adaptor(text_encoded, spk_embed) # (B=1,N,c)
        
        # variance adaptor and upsampling
        text_expanded, duration_prediction = self.variance_adaptor.infer(
            text_encoded,
            duration_rate=duration_rate,
        ) # (B=1,T,d)/(B=1,N)
        T = torch.sum(duration_prediction)
        assert T == text_expanded.size(1), "T={T}, text_expanded={text_expanded.size(1)}"
        
        # utterance adaptor
        text_expanded = self.utterance_adaptor(text_expanded, spk_embed) # (B=1,T,d)
        
        # mel decoding
        text_expanded = self.positional_encoding(text_expanded)
        _, mpg_output = self.mel_decoder(text_expanded) # text_expanded: (B=1,T,d)
        
        # postnet
        mel_output, _ = self.postnet(mpg_output)
        
        return (mel_output, duration_prediction) # (B=1,d,T)/(B=1,N)



if __name__ == "__main__":
    
    f = FasTTS()
    print(f)
    
    # spkid: (B,)
    # text_padding: (B,N,c)
    # duration_paddded: (B,N)
    # mel_padded: (B,d,T)
    s = torch.tensor((2,), dtype=torch.long)
    t = torch.randn(2,5,192)
    d = torch.tensor([[1., 2., 3, 2, 0], [1,1,1,1,1]], dtype=torch.float)
    m = torch.randn(2, 48, 8)
    y = f(s, t, d, m)
    print(y)
    