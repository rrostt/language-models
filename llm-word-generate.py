# train.py

import torch
import torch.nn as nn
import glob

from torchtext.datasets import WikiText2, WikiText103
data = WikiText2(root='data', split='train')
# loader = torch.utils.data.DataLoader(data, drop_last=True)

words = []
for i, text in enumerate(data):
    line = text.replace('  ', ' ')
    if len(line) > 0:
        words += list(filter(len, line.split(' '))) + ['\n']

# get unique characters in string text
vocab = tuple(set(words))
int2char = dict(enumerate(vocab))
char2int = {ch: ii for ii, ch in int2char.items()}

encoded = [char2int[ch] for ch in words]

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

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
  def __init__(self, optimizer, embed_size, warmup_steps, last_epoch=-1):
    self.warmup_steps = warmup_steps
    self.embed_size = embed_size
    self._step = 0
    super(WarmupScheduler, self).__init__(optimizer, last_epoch)

  def get_lr(self):
    self._step += 1
    return [self.embed_size ** -0.5 * min(self._step ** -0.5, self._step * self.warmup_steps ** -1.5) for group in self.optimizer.param_groups]

def load_checkpoint(filename, model, optim):
  checkpoint = torch.load(filename)
  model.load_state_dict(checkpoint['model'])
  optim.load_state_dict(checkpoint['optimizer'])
  loss = checkpoint['loss']
  epoch = checkpoint['epoch']
  return model, optim, loss, epoch

def load_latest(model, optim):
  files = glob.glob('ckpt/*.pt')
  files.sort()
  if len(files) > 0:
    model, optim, loss, epoch = load_checkpoint(files[-1], model, optim)
    print(f"Loaded checkpoint {files[-1]}")
    return model, optim, loss, epoch
  else:
    return model, optim, 0, -1

vocab_size = len(vocab)
embed_size = 512
depth = 12
heads = 8
max_length = 256
batch_size = 64
model = LLM(vocab_size, embed_size, depth, heads, max_length)

model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model size: {model_size / 1e6}M")

model.to('cuda')
optim = torch.optim.Adam(model.parameters(), lr=1e-5)
# scheduler = WarmupScheduler(optim, embed_size, 4000)
loss_fn = torch.nn.CrossEntropyLoss()

def generate(model, encoded_prompt, length):
#  model.eval()
  x = torch.tensor(encoded_prompt).unsqueeze(0).to('cuda')
  with torch.no_grad():
    for i in range(length):
      y = model(x[0, -max_length:])
      y = torch.softmax(y, dim=-1)
      y = torch.multinomial(y[0, -1, :], 1).unsqueeze(0)
      x = torch.cat([x, y], dim=1)
  return ' '.join([vocab[i] for i in x[0]]), ' '.join([vocab[i] for i in encoded_prompt])

def train(model, optim, loss_fn, data, epochs=10, device="cpu", start_epoch=0):
  model.to(device)
  print(len(data))
  lossi = []
  for epoch in range(start_epoch, start_epoch + epochs):
    for i in range((len(data) // max_length // batch_size) + 1):
      ix = torch.randint(0, len(data) - max_length - 1, (batch_size,)) if len(data) != max_length * batch_size + 1 else torch.zeros((batch_size,), dtype=torch.int64)
      x = torch.tensor([data[i:i + max_length] for i in ix]).to(device)
      y = torch.tensor([data[i + 1:i + max_length + 1] for i in ix]).to(device)
      if y.shape[1] != max_length:
        continue
      y_hat = model(x)
      loss = loss_fn(y_hat.view(-1, y_hat.shape[-1]), y.view(-1))
      lossi.append(loss.item())
      optim.zero_grad()
      loss.backward()
      optim.step()
      # scheduler.step()
      # if len(lossi)%100 == 0:
      #   print(f"epoch {epoch} i {i} loss {sum(lossi)/len(lossi)}")
    output, input = generate(model, data[:max_length], 12)
    output = output[len(input):]
    print(f"epoch {epoch} loss {sum(lossi)/len(lossi)}: {output}")
    save_latest(model, optim, sum(lossi)/len(lossi), epoch)
    lossi = []

model, optim, loss, last_epoch = load_latest(model, optim)
model.eval()

# train(model, optim, loss_fn, encoded, epochs=2000, device="cuda", start_epoch=last_epoch + 1)

output, input = generate(model, encoded[:max_length], 256)

print(output)
