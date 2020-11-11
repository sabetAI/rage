from transformers import (
    RagTokenizer,
    RagRetriever,
    RagTokenForGeneration,
    AdamW,
    get_linear_schedule_with_warmup,
)

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from argparse import ArgumentParser
from ipdb import set_trace
from tqdm import tqdm
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import difflib

import os
import json

from typing_extensions import Literal
import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

argparser = ArgumentParser()
argparser.add_argument("input")
argparser.add_argument("output")
argparser.add_argument("--eval", action="store_true")
argparser.add_argument("--fp16", action="store_true")
argparser.add_argument("--lr", type=float, default=1e-4)
argparser.add_argument("--adam_eps", type=float, default=1e-8)
argparser.add_argument("--batch_size", type=int, default=4)
argparser.add_argument("--warm_steps", type=int, default=0)
argparser.add_argument("--weight_decay", type=int, default=0)
argparser.add_argument("--num_epochs", type=int, default=2)
argparser.add_argument("--num_workers", type=int, default=4)
argparser.add_argument("--local_rank", type=int, default=0)
argparser.add_argument("--seed", type=int, default=0)
argparser.add_argument("--device", type=int, default=0)


args = argparser.parse_args()
torch.manual_seed(args.seed)

os.environ["WANDB_NOTEBOOK_NAME"] = "rage"
wandb.init(config=args, project="rage")
wandb.config["more"] = "nothing"


class LitRage(pl.LightningModule):
    def __init__(self, args, trainset, model):
        super().__init__()
        self.hparams.weight_decay = args.weight_decay
        self.hparams.learning_rate = args.lr
        self.hparams.adam_epsilon = args.adam_eps
        self.hparams.warmup_steps = args.warm_steps
        self.model = model
        self.total_steps = len(trainset)
        self.args = args

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output.loss.mean()
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    # def train_dataloader(self):
    #     return DataLoader(self.dataset["train"], batch_size=self.train_batch_size)


class RAGEDataset(Dataset):
    def __init__(self, args, tokenizer):
        super().__init__()
        corpus = [
            l.strip() for l in open(args.input, encoding="utf8").read().splitlines()
        ]
        self.examples = corpus
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        try:
            src, trg = self.examples[item].split("\t")
            return src, trg
        except:
            return None, None

    def collate(self, batch):
        src_sentences = [src for src, _ in batch if src is not None]
        trg_sentences = [trg for _, trg in batch if trg is not None]
        input_dict = tokenizer.prepare_seq2seq_batch(
            src_sentences,
            trg_sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return input_dict


def evaluate(model, dataloader, args):
    fout = open(args.output, "w", encoding="utf8")
    model.eval()
    model = model.to(args.device)

    with torch.no_grad():
        for input_batch in tqdm(dataloader):
            input_batch = {k: v.to(args.device) for k, v in input_batch.items()}
            logits = model(**input_batch)
            logits = logits[0]
            attention = input_batch["attention_mask"]
            argmax = [l[a == 1].softmax(1).max(1) for l, a in zip(logits, attention)]
            preds = [idx[val >= args.thresh] for val, idx in argmax]


if __name__ == "__main__":
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-base")
    trainset = RAGEDataset(args, tokenizer)
    retriever = RagRetriever.from_pretrained("facebook/rag-sequence-base")
    model = RagTokenForGeneration.from_pretrained(
        "facebook/rag-sequence-base", retriever=retriever
    )
    lit_rage = LitRage(args, trainset, model)

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=trainset.collate,
    )

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(
            args.output, "{epoch:02d}-{global_step:02d}-{val_loss:.2f}"
        ),
        verbose=True,
        save_top_k=-1,
        period=-1,
    )

    trainer = pl.Trainer(
        gpus=1,
        num_nodes=1,
        profiler=True,
        max_epochs=10,
        amp_level="O1",
        precision=16 if args.fp16 else 32,
        checkpoint_callback=checkpoint,
        accumulate_grad_batches=16,
    )

    trainer.fit(lit_rage, trainloader)
