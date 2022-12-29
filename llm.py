import torch
import torch.nn as nn

# (batch, seq, feature)
class PositionEncoding(torch.nn.Module):
  def __init__(self, max_length, embed_size):
    super(PositionEncoding, self).__init__()
    self.max_length = max_length
    self.embed_size = embed_size

    pos = torch.arange(0, max_length).unsqueeze(1)
    args = pos / (10000 ** (2 * torch.arange(0, embed_size, 2) / embed_size))
    self.pe = torch.zeros((max_length, embed_size))
    self.pe[:, ::2] = torch.sin(args)
    self.pe[:, 1::2] = torch.cos(args)

  def forward(self, x):
    self.pe = self.pe.to(x.device)
    return x + self.pe.unsqueeze(0)

# q, k, v: (batch, seq_len, embed_size)
def attention(q, k, v, mask=None):
  qk = q @ k.transpose(-1, -2) / (k.shape[-1] ** 0.5)
  if mask is not None:
    qk = qk + mask
  weights = torch.softmax(qk, dim=-1)
  return weights @ v

# (batch, seq, feature)
class MultiHeadAttention(torch.nn.Module):
  def __init__(self, embed_size, heads, max_length):
    super(MultiHeadAttention, self).__init__()
    self.embed_size = embed_size
    self.heads = heads
    self.head_size = embed_size // heads
    self.max_length = max_length

    self.mask = torch.zeros((max_length, max_length))
    ix = torch.triu_indices(max_length, max_length, 1)
    self.mask[ix[0], ix[1]] = -1e9

    self.Wq = torch.nn.Linear(embed_size, embed_size)
    self.Wk = torch.nn.Linear(embed_size, embed_size)
    self.Wv = torch.nn.Linear(embed_size, embed_size)

    self.Wo = torch.nn.Linear(embed_size, embed_size)

  def forward(self, x):
    b, s, e = x.shape
    self.mask = self.mask.to(x.device)
    # x is (batch, seq_len, embed_size)
    q = self.Wq(x).view(b, s, self.heads, self.head_size).transpose(1, 2).contiguous()
    k = self.Wk(x).view(b, s, self.heads, self.head_size).transpose(1, 2).contiguous()
    v = self.Wv(x).view(b, s, self.heads, self.head_size).transpose(1, 2).contiguous()
    x = attention(q, k, v, self.mask)
    # x is now (heads, seq_len, head_size)
    x = x.transpose(1, 2).contiguous().view(b, s, self.embed_size)
    return self.Wo(x)

# (batch, seq, feature)
class FeedForward(torch.nn.Module):
  def __init__(self, embed_size):
    super(FeedForward, self).__init__()
    self.main = torch.nn.Sequential(
      torch.nn.Linear(embed_size, embed_size * 4),
      torch.nn.ReLU(),
      torch.nn.Linear(embed_size * 4, embed_size)
    )

  def forward(self, x):
    return self.main(x)

class TransformerBlock(torch.nn.Module):
  def __init__(self, embed_size, heads, max_length):
    super(TransformerBlock, self).__init__()
    self.embed_size = embed_size
    self.heads = heads

    self.attention = MultiHeadAttention(embed_size, heads, max_length)
    self.norm1 = torch.nn.LayerNorm(embed_size)
    self.norm2 = torch.nn.LayerNorm(embed_size)
    self.ff = FeedForward(embed_size)

  def forward(self, x):
    attended = self.attention(x)
    x = self.norm1(attended + x)
    fed = self.ff(x)
    x = self.norm2(fed + x)
    return x

class LLM(torch.nn.Module):
  def __init__(self, vocab_size, embed_size, depth, heads, max_length):
    super(LLM, self).__init__()
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.max_length = max_length
    self.depth = depth
    self.heads = heads

    self.embedding = nn.Embedding(vocab_size, embed_size)
    self.pos_enc = PositionEncoding(max_length, embed_size)
    self.decoder = nn.Linear(embed_size, vocab_size)

    blocks = [TransformerBlock(embed_size, heads, max_length) for _ in range(depth)]
    self.blocks = nn.Sequential(*blocks)

    self.init_weights()

  def init_weights(self):
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_normal_(p)

  def forward(self, x):
    x = self.embedding(x)
    # position encoding
    x = self.pos_enc(x)
    # feed through transformer blocks
    x = self.blocks(x)
    out = self.decoder(x)
    return out # torch.softmax(out, dim=-1)
