# train.py

import torch
import torch.nn as nn
import glob
import os
import datetime

from transformers import GPT2Tokenizer

import json

from llm import LLM

from torchtext.datasets import WikiText2, WikiText103

# wiki text 2
# # load vocab from file if it exists
# # otherwise create vocab from words
# # and save
# vocab = []
# if os.path.exists('vocab.txt'):
#     with open('vocab.txt', 'r') as f:
#         for i, line in enumerate(f):
#             vocab.append(line.strip() if line != '\n' else '\n')
# else:
#   # create vocab from words
#   print('creating vocabulary...')
#   for i, word in enumerate(words):
#       if word not in vocab:
#           vocab.append(word)
#   # save vocab to file
#   with open('vocab.txt', 'w') as f:
#       for word in vocab:
#           f.write((word if word != '\n' else '') + '\n')

# int2char = dict(enumerate(vocab))
# char2int = {ch: ii for ii, ch in int2char.items()}

# encoded = [char2int[ch] for ch in words]

class WordTokenizer:
  def __init__(self, words):
    self.vocab = []
    if os.path.exists('vocab.txt'):
        with open('vocab.txt', 'r') as f:
            for i, line in enumerate(f):
                self.vocab.append(line.strip() if line != '\n' else '\n')
    else:
      # create vocab from words
      print('creating vocabulary...')
      for i, word in enumerate(words):
          if word not in self.vocab:
              self.vocab.append(word)
      # save vocab to file
      with open('vocab.txt', 'w') as f:
          for word in self.vocab:
              f.write((word if word != '\n' else '') + '\n')
    self.int2char = dict(enumerate(self.vocab))
    self.char2int = {ch: ii for ii, ch in self.int2char.items()}

  def encode(self, text):
    return [self.char2int[ch] for ch in text.split(' ')]
  
  def decode(self, tokens):
    return ' '.join([self.int2char[i] for i in tokens])

dataset = 'wikitext103'

data = WikiText103(root='data', split='train') if dataset == 'wikitext103' else WikiText2(root='data', split='train')

words = []
for i, text in enumerate(data):
    line = text.replace('  ', ' ')
    if len(line) > 0:
        words += list(filter(len, line.split(' '))) + ['\n']

# wikitext103
tokenizer = GPT2Tokenizer.from_pretrained('gpt2') if dataset == 'wikitext103' else WordTokenizer(words)

encoded = tokenizer.encode(' '.join(words))

def save_checkpoint(filename, model, optim, loss, epoch):
  torch.save({'optimizer': optim.state_dict(), 'model': model.state_dict(), 'loss': loss, 'epoch': epoch}, filename)

def load_checkpoint(filename, model, optim):
  checkpoint = torch.load(filename)
  model.load_state_dict(checkpoint['model'])
  optim.load_state_dict(checkpoint['optimizer'])
  loss = checkpoint['loss']
  epoch = checkpoint['epoch']
  return model, optim, loss, epoch

def save_latest(model, optim, loss, epoch):
  save_checkpoint(f'ckpt/e-{epoch}.pt', model, optim, loss, epoch)

def load_latest(model, optim):
  files = glob.glob('ckpt/*.pt')
  files.sort(key=os.path.getmtime)
  if len(files) > 0:
    model, optim, loss, epoch = load_checkpoint(files[-1], model, optim)
    print(f"Loaded checkpoint {files[-1]}")
    return model, optim, loss, epoch
  else:
    return model, optim, 0, -1

def generate(model, encoded_prompt, length):
  x = torch.tensor(encoded_prompt).unsqueeze(0).to(device)
  with torch.no_grad():
    for i in range(length):
      y = model(x[0, -max_length:])
      y = torch.softmax(y, dim=-1)
      y = torch.multinomial(y[0, -1, :], 1).unsqueeze(0)
      x = torch.cat([x, y], dim=1)
  return tokenizer.decode(x[0]), tokenizer.decode(encoded_prompt)

def train(model, optim, loss_fn, data, epochs=10, device="cpu", start_epoch=0):
  print(len(data))
  lossi = []
  step = 0
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
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optim.step()
      step += 1

      if len(lossi) % 128 == 0:
        output, input = generate(model, data[:max_length], 12)
        output = output[len(input):]
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{ts} epoch {epoch} step {step} loss {epoch_loss}: {output}", flush=True)
        save_latest(model, optim, epoch_loss, step)
        lossi = []

    output, input = generate(model, data[:max_length], 100)
    output = output[len(input):]
    epoch_loss = sum(lossi)/len(lossi)
    # print timestamp
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} epoch {epoch} \n\n{output}\n\n", flush=True)

device = torch.device('cuda')

# load config from config.json
# if it exists
# otherwise create config
# and save
config = {}
if os.path.exists('config.json'):
    with open('config.json', 'r') as f:
        config = json.load(f)
else:
    config['vocab_size'] = tokenizer.vocab_size # len(vocab)
    config['embed_size'] = 512
    config['depth'] = 12
    config['heads'] = 8
    config['max_length'] = 256
    config['batch_size'] = 64
    with open('config.json', 'w') as f:
        json.dump(config, f)

vocab_size = config['vocab_size']
embed_size = config['embed_size']
depth = config['depth']
heads = config['heads']
max_length = config['max_length']
batch_size = config['batch_size']
model = LLM(vocab_size, embed_size, depth, heads, max_length)

model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model size: {model_size / 1e6}M")

model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

last_epoch = -1
model, optim, loss, last_epoch = load_latest(model, optim)

if loss > 0:
  print(f'Previous loss {loss}')

train(model, optim, loss_fn, encoded, epochs=3000, device=device, start_epoch=last_epoch + 1)
