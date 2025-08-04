from tarfile import BLOCKSIZE
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 #32 # how many independent sequences will we process in parallel?
block_size = 256 #8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 #32 #the embedding dimensions #d_model
n_head = 6 #4
n_layer = 6 #3
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

class Head(nn.Module):
    """one head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) #(B,T,C)
        q = self.query(x) #(B,T,C)
        #compute attention scores ('affinities')
        wei = q@k.transpose(-2,-1) *C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) #(B,T,T) #No communication with past .'. this is decoder
        wei = F.softmax(wei, dim=1) # (B,T,T)
        #perform the weighted aggregation of the values
        v = self.value(x) # (B,T,c)
        out = wei@v #(B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) 
    def forward(self, x):
        #x: (B,T,C=n_embd=32)
        #h(x): (B,T,head_size=32)
        #O/p concatenated along last dimension (dim=-1) to yield shape: (B,T, num_heads*head_size)
        out = torch.cat([h(x) for h in self.heads], dim=-1) #o/p of self-attention itself
        #But we want to apply the projection back into the residual pathway
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),  #4*n_embd 'cause in paper, there is multiplication of 4 in the inside layers dimentionality(2048) as compared to the 512 dimentionality of the embeddings
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd) #projection layer going back into the residual pathway
        )
    def forward(self, x): #feedfwd is PER NODE/TOKEN basis. All tokens do this independently
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        #n_embd: embedding dimension, n_head:number of heads
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) #per token normalisation, makes row-wise i.e. 1 layer 1unit gaussian
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) #communication: affinity # x + ____  this is residual connection
        s = x + self.ffwd(self.ln2(x)) #computation: thinking
        return s

#super simple bigram model
class BigramLanguageModel(nn.Module):
    
    def __init__(self): #we need not pass around vocab_size as it is global variable
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #(6)Tokens for identity of the characters #n_embd =32, so, we've set a 32 dim embedding this time, unlike 65 (vocab_size) embedding. Dictonary size remmains 65 (vocab_size)
        #(6)Along with the identity encoding, we also encode the positions
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # # # self.sa_head = Head(n_embd) #1 self-attention Head of head_size =32
        # # self.sa_heads = MultiHeadAttention(num_heads=4, head_size=n_embd//4) #i.e. num_heads=4 heads of head_size=8-dimensional self_attention, so the inidivual head_size=8, and 4aro ka mila ke it becomes 4*8=32 
        # # self.ffwd = FeedForward(n_embd) #each node independently
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd), #final layer norm
        # )
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) #final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) #language Model Head, takes i/p n_embd and o/p vocab_size

    def forward(self, idx, targets=None):
        B,T = idx.shape
        #(1)idx and targets are both (B,T) tensor of integers
        #(2)logits = self.token_embedding_table(idx) # (B,T,C) C: channels, or embeddnig vector length
        #(3) This time, it wont give us logits directly, but letus call it token embeding
        tok_emb = self.token_embedding_table(idx)  #(5)(B,T,C) #---> this is embed C, i.e. n_embd = C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
        #(4) And let us introduce a simple linear layer to go from token emb to logits
        x = tok_emb + pos_emb #(B,T,C) #post_emb (T,C) becomes (1,T,C) then it is broadcasted and it becomes (B,T,C)
        # # x = self.sa_head(x) # (B,T,C) #apply one head of self-attention
        # x = self.sa_heads(x)      # 'Communication': gathered the affinity aabout the tokens
        # x = self.ffwd(x) #(B,T,C) # "Computation': now thinking about the tokens, each node/token independently
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) #(B,T,vocab_size) #(5) ---->This C and ^ this C are not the same. #and this is vocab_size


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
            #crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            #get the predictions
            logits, loss = self(idx_cond)
            #Since this is a bigram model, so focus only on the last time step, pluck out the last step as it will give the prediction for next step. Even though we're feeding all the history into the model, only last character is being used. History is not being used in bigram model rn.
            logits = logits[:, -1, :] # logits original size: (B,T,C), The last token's logits for each sequence in the batch, .'. becomes (B,C)
            #apply softmax to get probablities
            probs = F.softmax(logits, dim=-1) #(B,C)
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B,1) #For each batch row, we've 1 prediction for what comes next
            #append sampled indec to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1) i.e. concatenated along the 1st dim i.e. T, 0th dim is B here i.e. rows
        return idx
    
model = BigramLanguageModel()
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


    