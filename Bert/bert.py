import os
from pathlib import Path
import torch
import re
import random
import transformers, datasets
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import tqdm
from torch.utils.data import Dataset, DataLoader
import itertools
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam

MAX_LEN = 64


with open("movie_conversations.txt", "r" , encoding='iso-8859-1') as f:
    conv = f.readlines()

with open("movie_lines.txt", "r", encoding='iso-8859-1') as f:
    lines = f.readlines()

### splitting text using special lines
lines_dic = {}
for line in lines:
    objects = line.split(" +++$+++ ")
    lines_dic[objects[0]] = objects[-1]

### generate question answer pairs
pairs = []
for con in conv:
    ids = eval(con.split(" +++$+++ ")[-1])
    for i in range(len(ids)):
        qa_pairs = []
        
        if i == len(ids) - 1:
            break

        first = lines_dic[ids[i]].strip()  
        second = lines_dic[ids[i+1]].strip() 

        qa_pairs.append(' '.join(first.split()[:MAX_LEN]))
        qa_pairs.append(' '.join(second.split()[:MAX_LEN]))
        pairs.append(qa_pairs)

        # WordPiece tokenizer


### save data as txt file
if not os.path.exists('./data'):
    os.mkdir('./data')
text_data = []
file_count = 0

for sample in tqdm.tqdm([x[0] for x in pairs]):
    text_data.append(sample)

    # once we hit the 10K mark, save to file
    if len(text_data) == 10000:
        with open(f'./data/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))
        text_data = []
        file_count += 1

paths = [str(x) for x in Path('./data').glob('**/*.txt')]

### training own tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=True
)

tokenizer.train( 
    files=paths,
    vocab_size=30_000, 
    min_frequency=5,
    limit_alphabet=1000, 
    wordpieces_prefix='##',
    special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
    )

if not os.path.exists('./bert-it-1'):
    os.mkdir('./bert-it-1')
tokenizer.save_model('./bert-it-1', 'bert-it')


class BertDataset(Dataset):
    def __init__(self, tokenizer, data_pair, max_len=MAX_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_pair = data_pair

    def __len__(self):
        return len(self.data_pair)
    
    def __getitem__(self, idx):

        t1, t2, is_next = self.get_item(idx)

        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        t1 = [self.tokenizer.get_vocab().get("[CLS]")] + t1_random + [self.tokenizer.get_vocab().get('[SEP]')]
        t2 = t2_random + [self.tokenizer.get_vocab().get('[SEP]')]
        t1_label = [self.tokenizer.get_vocab().get("[PAD]")] + t1_label + [self.tokenizer.get_vocab().get("[PAD]")]
        t2_label = t2_label + [self.tokenizer.get_vocab().get("[PAD]")]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.max_len]
        bert_input = t1 + t2
        bert_label = (t1_label + t2_label)[:self.max_len]
        padding = [self.tokenizer.get_vocab().get("[PAD]") for _ in range(self.max_len - len(bert_input))]
        # bert_input.extend(padding)
        # bert_label.extend(padding)
        # segment_label.extend(padding)

        if len(bert_input) > self.max_len:
            bert_input = bert_input[:self.max_len]
            bert_label = bert_label[:self.max_len]
            segment_label = segment_label[:self.max_len]
        
        pad_token_id = self.tokenizer.get_vocab().get("[PAD]")  # Get the PAD token ID

        # Pad if shorter than max_len
        padding_length = self.max_len - len(bert_input)
        if padding_length > 0:
            bert_input.extend([pad_token_id] * padding_length)
            bert_label.extend([pad_token_id] * padding_length)
            segment_label.extend([pad_token_id] * padding_length) # Pad segment_label with PAD token ID too

        bert_label = torch.tensor(bert_label, dtype=torch.long)
        segment_label = torch.tensor(segment_label, dtype=torch.long)
        bert_input = torch.tensor(bert_input, dtype=torch.long)
        is_next = torch.tensor(is_next, dtype=torch.long)

        return {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next}
    
    def get_item(self, idx):
        t1, t2 = self.get_lines_from_corpus(idx)

        if random.random() < 0.5:
            return t1, t2, 1
        else:
            return t1, self.tokenizer.encode(self.data_pair[random.randrange(len(self.data_pair))][1]).ids, 0
        
    def get_lines_from_corpus(self, idx):
        # if idx >= len(self.data_pair):
        #     raise IndexError("Index out of range for data_pair")
        t1 = self.tokenizer.encode(self.data_pair[idx][0]).ids
        t2 = self.tokenizer.encode(self.data_pair[idx][1]).ids
        return t1, t2


    def random_word(self, tokens):
        output_label = []
        for i, token in enumerate(tokens):
            if random.random() < 0.15:
                output_label.append(token)
                if random.random() < 0.8:
                    tokens[i] = self.tokenizer.get_vocab().get('[MASK]')
                elif random.random() < 0.5:
                    tokens[i] = random.choice(list(self.tokenizer.get_vocab().values()))
            else:
                output_label.append(self.tokenizer.get_vocab().get('[PAD]'))
        return tokens, output_label
    
# class PositionEmbedding(torch.nn.Module):
#     def __init__(self, d_model, max_len=MAX_LEN):
#         super(PositionEmbedding, self).__init__()
#         self.d_model = d_model
#         self.max_len = max_len
        
#         # Create the positional encoding tensor
#         pe = torch.zeros(max_len, d_model).float()
#         pe.requires_grad = False
        
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
        
#         pe = pe.unsqueeze(0)
        
#         # Register 'pe' as a buffer. It will be automatically moved to the
#         # correct device when the model is moved.
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         # Return the positional embedding, sliced to the input's sequence length
#         return self.pe[:, :x.size(1)]

class PositionEmbedding(torch.nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super(PositionEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        for pos in range(self.max_len):
            for i in range(0, self.d_model, 2):
                pe[pos, i] = math.sin( pos / np.power(10000, (2 * i)/self.d_model))
                pe[pos, i + 1] = math.cos( pos / np.power(10000, (2 * (i + 1))/self.d_model))
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe
    

class BertEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, max_len=MAX_LEN):
        super(BertEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = embed_size
        self.max_len = max_len
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.segment = torch.nn.Embedding(3, embed_size, padding_idx=0)  # 3 segments: [CLS], [SEP], [PAD]
        self.position = PositionEmbedding(embed_size, max_len)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, input_ids, segment_ids):
        # The positional embedding from self.position(input_ids) is now correctly
        # on the GPU and sliced to the right sequence length.
        x = self.token(input_ids) + self.position(input_ids) + self.segment(segment_ids)
        return self.dropout(x)
    


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(0.1)

    def attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_model)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        scores = torch.matmul(scores, value)
        return scores
    
    def multiattention(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = self.query(query).view(batch_size, -1, self.num_heads, self.d_model// self.num_heads).permute(0, 2, 1, 3)
        key = self.key(key).view(batch_size, -1, self.num_heads, self.d_model// self.num_heads).permute(0, 2, 1, 3)
        value = self.value(value).view(batch_size, -1, self.num_heads, self.d_model// self.num_heads).permute(0, 2, 1, 3)

        # if mask is not None:
            # mask = mask.unsqueeze(1).unsqueeze(2)
            # mask = mask.repeat(1, self.num_heads, 1, 1)

        scores = self.attention(query, key, value, mask)

        scores = scores.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)

        scores = self.output(scores)
        return scores
    
    def forward(self, query, key, value, mask=None):
        return self.multiattention(query, key, value, mask)
        
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.feedforward = FeedForward(d_model, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, embedding, mask=None):
        interacted = self.dropout1(self.multihead_attention(
            embedding, embedding, embedding, mask))
        interacted = self.layernorm1(embedding + interacted)
        feed_forwarded = self.dropout1(self.feedforward(interacted))
        encoded = self.layernorm1(interacted + feed_forwarded)
        return encoded
    
class BERT(torch.nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, d_model=768, n_layers=12, heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads

        # paper noted they used 4 * hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = d_model * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BertEmbedding(vocab_size=vocab_size, embed_size=d_model)

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = torch.nn.ModuleList(
            [EncoderLayer(d_model, heads, d_model * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info):
        # attention masking for padded token
        # (batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x
         

class NextSentencePrediction(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(NextSentencePrediction, self).__init__()
        self.linear = nn.Linear(d_model, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
    
class MLM(nn.Module):
    def __init__(self,hidden_size, vocab_size):
        super(MLM, self).__init__()
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.activation = nn.LogSoftmax(dim=-1)
    def forward(self, x):
        x = self.linear(x)
        return self.activation(x)
    
class BERTLM(nn.Module):
    def __init__(self, bert: BERT, vocab_size):
        super(BERTLM, self).__init__()
        self.bert = bert
        self.vocab_size = vocab_size
        self.nsp = NextSentencePrediction(self.bert.d_model)
        self.mlm = MLM(self.bert.d_model, self.vocab_size)

    def forward(self, x, segment_labels):
        x = self.bert(x, segment_labels)
        return self.nsp(x[:,0]), self.mlm(x)
    
class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

class BERTTrainer:
    def __init__(
        self, 
        model, 
        train_dataloader, 
        test_dataloader=None, 
        lr= 1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        warmup_steps=10000,
        log_freq=10,
        device='cuda'
        ):

        self.device = device
        self.model = model
        self.model.to(self.device)
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(
            self.optim, self.model.bert.d_model, n_warmup_steps=warmup_steps
            )

        # CORRECT: Use CrossEntropyLoss for NSP (takes logits)
        self.nsp_criterion = torch.nn.CrossEntropyLoss()
        # CORRECT: Use NLLLoss for MLM (takes log-probabilities)
        self.mlm_criterion = torch.nn.NLLLoss(ignore_index=0)
        
        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
    
    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        
        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        
        mode = "train" if train else "test"

        # progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])

            # 2-1. CORRECTED: Use the new NSP criterion for the next sentence loss
            next_loss = self.nsp_criterion(next_sent_output, data["is_next"])

            # 2-2. CORRECTED: Use the MLM criterion for the masked token loss
            mask_loss = self.mlm_criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

            # 2-3. Adding next_loss and mask_loss
            loss = next_loss + mask_loss

            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # next sentence prediction accuracy
            correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_next"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        print(
            f"EP{epoch}, {mode}: \
            avg_loss={avg_loss / len(data_iter)}, \
            total_acc={total_correct * 100.0 / total_element}"
        ) 

'''test run'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = BertDataset(
   data_pair=pairs, max_len=MAX_LEN, tokenizer=tokenizer)

train_loader = DataLoader(
   train_data, batch_size=32, shuffle=True, pin_memory=True)

bert_model = BERT(
  vocab_size=tokenizer.get_vocab_size(),
  d_model=768,
  n_layers=2,
  heads=12,
  dropout=0.1
)

bert_lm = BERTLM(bert_model, tokenizer.get_vocab_size())
bert_trainer = BERTTrainer(bert_lm, train_loader, device=device)
epochs = 20

for epoch in range(epochs):
    bert_trainer.train(epoch)

torch.save(bert_lm.state_dict(), 'bert_lm.pth')