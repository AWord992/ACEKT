import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
# from .utils import transformer_FFN, ut_mask, pos_encode, get_clones
from torch.nn import LayerNorm
# from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy
from .cim import CIM, CIMConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def telu(input):
    return input * torch.tanh(torch.exp(input))

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class aceKT(nn.Module):
    def __init__(self, n_question, n_pid, 
            d_model, n_blocks, dropout, dropout_pred, dropout_freq, use_ssm, use_freq, use_telu=1, use_tfm=1,
            d_ff=256, kernel_size=4, seq_len=200,
            kq_same=1, final_fc_dim=512, final_fc_dim2=256, num_attn_heads=8, separate_qa=False, emb_type="qid", emb_path=""):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "acekt"
        print(f"model_name: {self.model_name}, emb_type: {emb_type}")
        self.n_question = n_question
        self.dropout_pred = dropout_pred
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        embed_l = d_model
        if self.n_pid > 0:
            if emb_type.find("scalar") != -1:
                # print(f"question_difficulty is scalar")
                self.difficult_param = nn.Embedding(self.n_pid+1, 1) # Question difficulty
            else:
                self.difficult_param = nn.Embedding(self.n_pid+1, embed_l) # Question difficulty
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l) # question emb, Summarizes the changes in problems (questions) that include the current question (concept)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l) # interaction emb
        
        if emb_type.startswith("qid"):
            # n_question+1 ,d_model
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa: 
                    self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
            else: # false default
                self.qa_embed = nn.Embedding(2, embed_l)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads,
                                  dropout=dropout, dropout_freq = dropout_freq,
                                  d_model=d_model, kernel_size=kernel_size, d_ff=d_ff,  kq_same=self.kq_same, seq_len=seq_len,
                                  use_ssm=use_ssm, use_freq=use_freq, use_telu=use_telu, use_tfm=use_tfm)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout_pred),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout_pred),
            nn.Linear(final_fc_dim2, 1)
        )

        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  d_model# c_ct
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.qa_embed(target)+q_embed_data
        return q_embed_data, qa_embed_data

    def get_attn_pad_mask(self, sm):
        batch_size, l = sm.size()
        pad_attn_mask = sm.data.eq(0).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.expand(batch_size, l, l)
        return pad_attn_mask.repeat(self.nhead, 1, 1)

    def forward(self, dcur, qtest=False, train=False):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        q, c, r, qshft, cshft, rshft = q.to(device), c.to(device), r.to(device), qshft.to(device), cshft.to(device), rshft.to(device)
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        q_data = torch.cat((c[:,0:1], cshft), dim=1)
        target = torch.cat((r[:,0:1], rshft), dim=1)

        emb_type = self.emb_type

        # Batch First
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        if self.n_pid > 0 and emb_type.find("norasch") == -1: # have problem id
            if emb_type.find("aktrasch") == -1:
                q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct summerizes the changes in problems (questions) that include the current question (concept)
                pid_embed_data = self.difficult_param(pid_data)  # uq The difficulty of the current problem
                q_embed_data = q_embed_data + pid_embed_data * \
                    q_embed_diff_data  # uq *d_ct + c_ct # question encoder

            else:
                q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct summarizes the changes in problems (questions) that include the current question (concept)
                pid_embed_data = self.difficult_param(pid_data)  # uq The difficulty of the current problem
                q_embed_data = q_embed_data + pid_embed_data * \
                    q_embed_diff_data  # uq *d_ct + c_ct # question encoder

                qa_embed_diff_data = self.qa_embed_diff(
                    target)  # f_(ct,rt) or #h_rt (qt, rt) Difference Vector
                qa_embed_data = qa_embed_data + pid_embed_data * \
                        (qa_embed_diff_data+q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）

        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        y2, y3 = 0, 0
        if emb_type in ["qid", "qidaktrasch", "qid_scalar", "qid_norasch"]:
            d_output = self.model(q_embed_data, qa_embed_data)

            concat_q = torch.cat([d_output, q_embed_data], dim=-1)
            # print(f"concat_q shape: {concat_q.shape}")
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)

        if train:
            return preds, y2, y3
        else:
            if qtest:
                return preds, concat_q
            else:
                return preds

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation)

    def forward(self, x):
        return self.conv(x)[:, :, :-(self.conv.padding[0])] 

class RhythmPerceptionModule(nn.Module):
    def __init__(self, dropout, hidden_size, kernel_size):
        super(RhythmPerceptionModule, self).__init__()
        self.out_dropout = nn.Dropout(dropout)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.causal_conv = CausalConv1d(hidden_size, hidden_size, kernel_size)
        self.c = kernel_size // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape

        # Causal Convolution
        input_tensor = input_tensor.permute(0, 2, 1)  # switch to [batch, hidden, seq_len]
        low_pass = self.causal_conv(input_tensor)
        low_pass = low_pass.permute(0, 2, 1)  # back to [batch, seq_len, hidden]

        # High-pass filter
        high_pass = input_tensor.permute(0, 2, 1) - low_pass  # switch to [batch, seq_len, hidden]
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor.permute(0, 2, 1))

        return hidden_states

class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, kernel_size,
                 d_ff, n_heads, dropout, dropout_freq, kq_same, seq_len, use_ssm, use_freq, use_telu, use_tfm):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.use_freq = use_freq      


        self.blocks_2 = nn.ModuleList([
            CascadedBlock(d_model=d_model,d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, 
                            dropout_freq=dropout_freq, kernel_size=kernel_size, use_ssm=use_ssm, use_freq=use_freq, use_telu=use_telu, use_tfm=use_tfm)
            for _ in range(n_blocks)
        ])
        self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)

    def forward(self, q_embed_data, qa_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        # position encoding
        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb
        qa_posemb = self.position_emb(qa_embed_data)
        qa_embed_data = qa_embed_data + qa_posemb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed


        # encoder
        for block in self.blocks_2:
            x = block(x, y)
  
        return x

class CascadedBlock(nn.Module):
    def __init__(self, d_model,d_ff, n_heads, dropout, 
                kq_same, dropout_freq, kernel_size, use_freq, use_ssm, use_telu, use_tfm):
        super().__init__()
        self.use_ssm = use_ssm
        self.use_freq = use_freq
        self.use_tfm = use_tfm

        if use_tfm == 1:
            self.transformer = TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
        
        if self.use_ssm == 1:
            cimConfig = CIMConfig(d_model=d_model, n_layers=1, d_conv=4, use_telu=use_telu)
            self.cim = CIM(cimConfig)

        if self.use_freq == 1:
            self.filter_layer = RhythmPerceptionModule(dropout_freq, d_model, kernel_size)  
    
    def forward(self, x, y):
        if self.use_freq == 1:
            x = self.filter_layer(x)
            y = self.filter_layer(y)
        if self.use_tfm == 1:
            x = self.transformer(mask=0, query=x, key=x, values=y, apply_pos=True)
        output = x
        
        if self.use_ssm == 1:
            output = self.cim(x)

        return output  

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1        
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)

        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True) # Only previous information can be seen, and current information cannot be seen. At this time, the first row of scores will be set to 0, indicating that the historical interaction information cannot be seen for the first question.
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2)) 
        query = self.layer_norm1(query) # layer norm

        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))

            query = query + self.dropout2((query2)) # Residual Connection
            query = self.layer_norm2(query) # lay norm

        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: Dimensions of each head
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"before zero pad scores: {scores.shape}")
    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # The score of the first row is set to 0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)
