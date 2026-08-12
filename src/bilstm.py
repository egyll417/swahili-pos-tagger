"""Char-level BiLSTM POS tagger. SEEDED for reproducibility."""
import torch, torch.nn as nn, torch.optim as optim, random, numpy as np, argparse
from reader import read_conll
from vocab import build_char_vocab, build_tag_vocab, word_to_char_indices, PAD_TOKEN

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def sentence_to_tensors(sentence, char2idx, tag2idx):
    words=[w for w,_ in sentence]; tags=[t for _,t in sentence]
    cid=[word_to_char_indices(w,char2idx) for w in words]
    lens=[len(x) for x in cid]; mx=max(lens); pad=char2idx[PAD_TOKEN]
    padded=[x+[pad]*(mx-len(x)) for x in cid]
    return (torch.tensor(padded,dtype=torch.long),
            torch.tensor(lens,dtype=torch.long),
            torch.tensor([tag2idx.get(t,0) for t in tags],dtype=torch.long))

class CharBiLSTMTagger(nn.Module):
    def __init__(self, char_vocab_size, num_tags, char_emb_dim=32, char_hidden_dim=64,
                 word_hidden_dim=128, pad_idx=0):
        super().__init__()
        self.char_embedding=nn.Embedding(char_vocab_size,char_emb_dim,padding_idx=pad_idx)
        self.char_lstm=nn.LSTM(char_emb_dim,char_hidden_dim,batch_first=True,bidirectional=True)
        self.word_lstm=nn.LSTM(char_hidden_dim*2,word_hidden_dim,batch_first=True,bidirectional=True)
        self.classifier=nn.Linear(word_hidden_dim*2,num_tags)
    def forward(self,char_ids,word_lengths):
        emb=self.char_embedding(char_ids)
        packed=nn.utils.rnn.pack_padded_sequence(emb,word_lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,(h_n,_)=self.char_lstm(packed)
        wr=torch.cat([h_n[0],h_n[1]],dim=-1).unsqueeze(0)
        out,_=self.word_lstm(wr)
        return self.classifier(out.squeeze(0))

def train_model(model,train_sents,char2idx,tag2idx,epochs=15,lr=1e-3):
    opt=optim.Adam(model.parameters(),lr=lr); loss_fn=nn.CrossEntropyLoss()
    sents=list(train_sents)
    for ep in range(epochs):
        random.shuffle(sents); tot=0.0
        for s in sents:
            c,l,t=sentence_to_tensors(s,char2idx,tag2idx)
            opt.zero_grad(); loss=loss_fn(model(c,l),t); loss.backward(); opt.step()
            tot+=loss.item()
        print(f"  epoch {ep}: avg loss = {tot/len(sents):.4f}")
    return model

def generate_predictions_file(sents,char2idx,tag2idx,model,out):
    id2tag={v:k for k,v in tag2idx.items()}; model.eval()
    with torch.no_grad(), open(out,'w',encoding='utf-8') as f:
        for s in sents:
            c,l,_=sentence_to_tensors(s,char2idx,tag2idx)
            pred=torch.argmax(model(c,l),dim=-1)
            for (w,_),p in zip(s,pred): f.write(f"{w} {id2tag[p.item()]}\n")
            f.write("\n")
    model.train(); print(f"  wrote {out}")

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=1)
    ap.add_argument('--epochs',type=int,default=15)
    ap.add_argument('--out-prefix',default='../output/bilstm')
    a=ap.parse_args()
    set_seed(a.seed)
    train=read_conll('../data/train.txt')
    char2idx=build_char_vocab(train); tag2idx=build_tag_vocab(train)
    print(f"[bilstm] seed={a.seed} chars={len(char2idx)} tags={len(tag2idx)}")
    model=CharBiLSTMTagger(len(char2idx),len(tag2idx))
    train_model(model,train,char2idx,tag2idx,epochs=a.epochs)
    torch.save(model.state_dict(),f"{a.out_prefix}_model.pt")
    for split in ['dev','test']:
        generate_predictions_file(read_conll(f'../data/{split}.txt'),char2idx,tag2idx,model,
                                  f"{a.out_prefix}_{split}_pred.txt")
