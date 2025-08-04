import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

#------------
torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding ='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch: i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] #encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x, y

#context manager - it tells torch, that everything happens inside this function, we wont call .backward on,
#so pytorch is more efficient with its memory use, wont call intermediate variables etc

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() #setting the model to eval
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() #resetting the model back to train
    return out

#super simple bigram model
class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        #idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C) C: channels, or embeddnig vector length

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is (B,T) array of the B no. of current context-s (of length T)
        #job of generate is to extend the (B,T) to (B, T+1) to (B, T+2) and so on
        #i.e. for all rows in a Batch (i.e. B =4) the columns inc from T to T+1, T+1 is then fed back to generate (T+1)+1 = T+2 and this goes on till the number of max_tookens that we want
        for _ in range(max_new_tokens):
            #get the predictions
            logits, loss = self(idx)
            #Since this is a bigram model, so focus only on the last time step, pluck out the last step as it will give the prediction for next step. Even though we're feeding all the history into the model, only last character is being used. History is not being used in bigram model rn.
            logits = logits[:, -1, :] # logits original size: (B,T,C), The last token's logits for each sequence in the batch, .'. becomes (B,C)
            #apply softmax to get probablities
            probs = F.softmax(logits, dim=-1) #(B,C)
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B,1) #For each batch row, we've 1 prediction for what comes next
            #append sampled indec to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1) i.e. concatenated along the 1st dim i.e. T, 0th dim is B here i.e. rows
        return idx
    
model = BigramLanguageModel(vocab_size)
m = model.to(device) #move model params to device
print(f'Running on device: {device}')

#create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    #every once in a while evaluate the loss on train and val sets
    if iter%eval_interval == 0:
        losses = estimate_loss() #it averages the loss over multiple batches
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    #sample a batch of data
    xb,yb = get_batch('train')

    #evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


    