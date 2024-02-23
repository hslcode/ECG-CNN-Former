import torch
import numpy as np
import torch.nn as nn
import math
from CNN import CNN
# Transformer Parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
d_model = 28  # 频率所在维度
d_ff = 128  # FeedForward dimension ，将attention特征映射到高维度空间
d_k = d_v = 7  # dimension of K(=Q), V（Q和K的维度需要相同，这里为了方便让K=V）需满足：n_heads*d_k=d_model
n_layers = 4  # number of Encoder of Decoder Layer（Block的个数）
n_heads = 4  # number of heads in Multi-Head Attention（有几套头）
d_class = 5 #分类种类数量
# ====================================================================================================
# Transformer模型
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        #unsqueeze(1)在第二个维度插入一个维度，(2,)->(2,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe: [seq_len, batch_size, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x.cuda()
        pe_encode = self.pe[:x.size(0), :]
        c = pe_encode.cpu().numpy()
        a = x.cpu().numpy()
        x = x + pe_encode
        b = x.cpu().numpy()
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    # pad mask的作用：在对value向量加权平均的时候，可以让pad对应的alpha_ij=0，这样注意力就不会考虑到pad向量
    """这里的q,k表示的是两个序列（跟注意力机制的q,k没有关系），例如encoder_inputs (x1,x2,..xm)和encoder_inputs (x1,x2..xm)
    encoder和decoder都可能调用这个函数，所以seq_len视情况而定
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()  # 这个seq_q只是用来expand维度的
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # 例如:seq_k = [[1,2,3,4,0], [1,2,3,5,0]]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k] 构成一个立方体(batch_size个这样的矩阵)
def get_attn_subsequence_mask(seq):
    """建议打印出来看看是什么的输出（一目了然）
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


# ==========================================================================================
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    #1......................................................
    #def forward(self, Q, K, V, attn_mask):
    def forward(self, Q, K, V):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        说明：在encoder-decoder的Attention层中len_q(q1,..qt)和len_k(k1,...km)可能不同
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # mask矩阵填充scores（用-1e9填充scores中与attn_mask中值为1位置相对应的元素）
        #1........................................................................
        #scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)  # 对最后一个维度(v)做softmax
        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        context = torch.matmul(attn, V)  # context: [batch_size, n_heads, len_q, d_v]
        # context：[[z1,z2,...],[...]]向量, attn注意力稀疏矩阵（用于可视化的）
        return context, attn


class MultiHeadAttention(nn.Module):
    """这个Attention类可以实现:
    Encoder的Self-Attention
    Decoder的Masked Self-Attention
    Encoder-Decoder的Attention
    输入：seq_len x d_model
    输出：seq_len x d_model
    """
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)  # q,k必须维度相同，不然无法做点积
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        # 这个全连接层可以保证多头attention的输出仍然是seq_len x d_model
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    #1..........................................................................
    #def forward(self, input_Q, input_K, input_V, attn_mask):
    def forward(self, input_Q, input_K, input_V):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # 下面的多头的参数矩阵是放在一起做线性变换的，然后再拆成多个头，这是工程实现的技巧
        # B: batch_size, S:seq_len, D: dim
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, Head, W) -trans-> (B, Head, S, W)
        #           线性变换               拆成多头

        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k] # K和V的长度一定相同，维度可以不同
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # 因为是多头，所以mask矩阵要扩充成4维的
        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        #1....................................................................................
        #attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        #1...................................................................................
        #context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context, attn = ScaledDotProductAttention()(Q, K, V)
        # 下面将不同头的输出向量拼接在一起
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)

        # 这个全连接层可以保证多头attention的输出仍然是seq_len x d_model
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to(device)(output + residual), attn


# Pytorch中的Linear只会对最后一维操作，所以正好是我们希望的每个位置用同一个全连接网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    #1................................................
    #def forward(self, enc_inputs, enc_self_attn_mask):
    def forward(self, enc_inputs):
        """E
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]  mask矩阵(pad mask or sequence mask)
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        # 第一个enc_inputs * W_Q = Q
        # 第二个enc_inputs * W_K = K
        # 第三个enc_inputs * W_V = V
        #1..........................................................
        # enc_inputs to same Q,K,V（未线性变换前）

        #1....................................................................................
        #enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,enc_self_attn_mask)
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)

        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #self.src_emb = nn.Embedding(src_vocab_size, d_model)  # token Embedding
        self.pos_emb = PositionalEncoding(d_model)  # Transformer中位置编码时固定的，不需要学习
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        """
        #enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = enc_inputs
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        # Encoder输入序列的pad mask矩阵

        #1...........................................................................................
        #enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]

        enc_self_attns = []  # 在计算中不需要用到，它主要用来保存你接下来返回的attention的值（这个主要是为了你画热力图等，用来看各个词之间的关系
        for layer in self.layers:  # for循环访问nn.ModuleList对象
            # 上一个block的输出enc_outputs作为当前block的输入
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention

            #1...............................................................
            #enc_outputs, enc_self_attn = layer(enc_outputs,enc_self_attn_mask)
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)  # 这个只是为了可视化
        return enc_outputs, enc_self_attns


class My_Transformer(nn.Module):
    def __init__(self,device):
        super(My_Transformer, self).__init__()
        self.encoder = Encoder().to(device)

        '''
            self.fc = nn.Sequential(
            nn.Linear(3904, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 64, bias=False)
            nn.ReLU(),
            nn.Linear(64, d_class, bias=False)
        )
        '''
        self.projection1 = nn.Linear(868, 2048, bias=False).to(device)#(868=fre*time)
        self.ReLU = nn.ReLU().to(device)
        self.projection2 = nn.Linear(2048, 512, bias=False).to(device)
        self.projection3 = nn.Linear(512, d_class, bias=False).to(device)
        #self.softmax = nn.Softmax(dim=0).to(device)
    def forward(self, enc_inputs):

        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        enc_outputs = torch.flatten(enc_outputs, start_dim=1)

        dec_logits1 = self.projection1(enc_outputs)
        dec_logits1 = self.ReLU(dec_logits1)

        dec_logits2 = self.projection2(dec_logits1)
        dec_logits2 = self.ReLU(dec_logits2)

        dec_logits3 = self.projection3(dec_logits2)
        dec_logits3 = self.ReLU(dec_logits3)
        dec_logits = dec_logits3
        #dec_logits = self.softmax(dec_logits3)
        return dec_logits.view(-1, dec_logits.size(-1))
        #return dec_logits.view(-1), enc_self_attns
class Net(nn.Module):
    def __init__(self,device):
        super(Net, self).__init__()
        self.encoder = Encoder().to(device)
        self.CNN_Net = CNN()
        self.head = nn.Sequential(

            nn.Linear(1668, 2048, bias=True),
            nn.LayerNorm(2048),
            nn.ReLU(),

            # nn.LayerNorm(1),
            nn.Linear(2048, 512, bias=True),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 5, bias=True),
        )
    def forward(self, Signal_TF,Signal_org):
        enc_outputs, enc_self_attns = self.encoder(Signal_TF)
        enc_outputs = torch.flatten(enc_outputs, start_dim=1)
        cnn_outputs = self.CNN_Net(Signal_org)
        cnn_outputs = cnn_outputs.view(cnn_outputs.size(0),-1)
        net_outputs = torch.cat([enc_outputs,cnn_outputs],dim=1)
        outputs = self.head(net_outputs)
        return outputs

