from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from argparse import ArgumentParser
from ipdb import set_trace
from tqdm import tqdm
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import difflib

import wandb


argparser = ArgumentParser()
argparser.add_argument('input')
argparser.add_argument('output')
argparser.add_argument('--eval', action='store_true')
argparser.add_argument('--lr', type=float, default=1e-4)
argparser.add_argument('--adam_eps', type=float, default=1e-8)
argparser.add_argument('--batch_size', type=int, default=48)
argparser.add_argument('--warm_steps', type=int, default=0)
argparser.add_argument('--num_epochs', type=int, default=2)
argparser.add_argument('--num_workers', type=int, default=0)
argparser.add_argument('--weight', type=float, default=4.0)
argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument('--device', type=int, default=0)

args = argparser.parse_args()

torch.manual_seed(args.seed)

wandb.init(config=args, project="ctrl-cls")
wandb.config["more"] = "nothing"

class RAGDataset(Dataset):
    def __init__(self, args, tokenizer):
        super().__init__()
        if args.eval:
            corpus = [l.strip() for l in open(args.input, encoding='utf8').read().splitlines()]
        else:
            corpus = [l.strip() for l in open(args.input, encoding='utf8').read().splitlines()]
        self.examples = corpus
        self.tokenizer = tokenizer
        self.eval = args.eval
        self.args = args

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        if self.eval:
            text = self.examples[item]
            return text
        else:
            src, trg = self.examples[item].split('\t')
            return src, trg

    def collate(self, batch):
        if self.eval:
            input_ids = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            return input_ids
        else:
            src_sentences = [src for src,_ in batch]
            trg_sentences = [trg for _,trg in batch]
            input_dict = tokenizer.prepare_seq2seq_batch(src_sentences, trg_sentences, padding=True, truncation=True, return_tensors="pt")

            return input_dict

def get_optimizer(model, args):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        eps=args.adam_eps,
    )

    return optimizer

def get_scheduler(dataloader, optimizer, args):
    t_total = (
     (
         len(dataloader.dataset)
         // (args.batch_size)
     )
     // 1
     * float(args.num_epochs)
    )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warm_steps,
        num_training_steps=t_total,
    )
    return lr_scheduler

class Printer:
    def __init__(self, interval):
        self.interval = interval
        self.ctr = 0

    def print(self, preds, labels):
        if self.ctr % self.interval == 0:
            print(labels[0])
            print(preds[0])
        self.ctr += 1

def train(model, dataloader, args):
    model.train()
    model = model.to(args.device)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_scheduler(dataloader, optimizer, args)
    printer = Printer(interval=10)

    for epoch in range(args.num_epochs):
        for batch_input in tqdm(dataloader):
            optimizer.zero_grad()
            batch_dict = {k:v.to(args.device) for k,v in batch_input.items()}
            output = model(**batch_dict)
            loss = output.loss.mean()
            print(loss.item())
            wandb.log({"epoch": epoch, "loss": loss.item()})
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        checkpt = f"{args.output}-checkpoint{epoch}"
        model.save_pretrained(checkpt)
    

def evaluate(model, dataloader, args):
    fout = open(args.output, 'w', encoding='utf8')
    model.eval()
    model = model.to(args.device)
    
    with torch.no_grad():
        for input_batch in tqdm(dataloader):
            input_batch = {k:v.to(args.device) for k,v in input_batch.items()}
            logits = model(**input_batch)
            logits = logits[0]
            attention = input_batch["attention_mask"]
            argmax = [l[a == 1].softmax(1).max(1) for l,a in zip(logits, attention)]
            preds = [idx[val >= args.thresh] for val, idx in argmax]
            
            for pred in preds:
                if args.multi:
                    output_tags = pred[pred != 0].tolist()
                    errant_tags = [errant_map[o] for o in set(output_tags)]
                    if len(errant_tags):
                        output = ', '.join(errant_tags) + '\n'
                    else:
                        output = 'noop\n'
                else:
                    output = errant_map[pred] + '\n'
                fout.write(output)

if __name__ == '__main__':
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-base")
    retriever = RagRetriever.from_pretrained("facebook/rag-sequence-base")
    model = RagTokenForGeneration.from_pretrained("facebook/rag-sequence-base", retriever=retriever)
    set_trace()

    dataset = RAGDataset(args, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=dataset.collate) 

    if args.eval:
        evaluate(model, dataloader, args)
    else:
        train(model, dataloader, args)
