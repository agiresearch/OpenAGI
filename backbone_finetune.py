import torch
from torch import optim
from torch.optim import optimizer
from torch.utils.data import Dataset, DataLoader
from utils import set_seed, Logger, construct_optimizer
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
from generate_trie import generate_sequence, candidates
from undecorated import undecorated
from types import MethodType
import random
import json


def load_data(args):
    with open(args.prompt_dir, "r") as f:
        prompts = f.read()
    prompts = prompts.split("\n")[:-1]

    with open(args.answer_dir, "r") as f:
        answers = f.read()
    answers = answers.split("\n")[:-1]

    data = [(prompt, answer) for prompt, answer in zip(prompts, answers)]
    # random.shuffle(data)
    with open("finetune_train.json", "w") as f:
        json.dump(data[: args.num_train], f)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    class InputDataset(Dataset):
        def __init__(self, data):
            super().__init__()
            self.prompts = [d[0] for d in data]
            self.answers = [d[1] for d in data]

        def __len__(self):
            return len(self.answers)

        def __getitem__(self, index):
            prompts = tokenizer(self.prompts[index], return_tensors="pt")
            answers = tokenizer(self.answers[index], return_tensors="pt")

            return (
                prompts["input_ids"].squeeze(),
                prompts["attention_mask"].squeeze(),
                answers["input_ids"].squeeze(),
            )

    train_dataset = InputDataset(data[: args.num_train])
    test_dataset = InputDataset(data[args.num_train :])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader


def train(args, logger):
    logger.log("load model")
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(args.device)
    generate_with_grad = undecorated(model.generate)
    model.generate_with_grad = MethodType(generate_with_grad, model)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    train_loader, test_loader = load_data(args)
    optimizer, scheduler = construct_optimizer(args, model, 20)

    logger.log("start training")
    model.zero_grad()
    best_accuracy = 0
    for e in range(args.epochs):
        logger.log("start epoch {}".format(e))
        for batch in train_loader:
            input_ids = batch[0].to(args.device)
            attn_mask = batch[1].to(args.device)
            answer_ids = batch[2].to(args.device)
            loss = model(
                input_ids=input_ids, attention_mask=attn_mask, labels=answer_ids
            ).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        logger.log("***** start evaluate in epoch {} *****".format(e))
        total = 0
        accurate = 0
        questions = []
        predictions = []
        targets = []
        for batch in test_loader:
            input_ids = batch[0].to(args.device)
            attn_mask = batch[1].to(args.device)
            answer_ids = batch[2][0].tolist()
            predicted_sequence, _ = generate_sequence(
                input_ids=input_ids,
                model=model,
                tokenizer=tokenizer,
                candidates=candidates,
                module_length=5,
                num_beams=5,
                num_return_sequences=args.num_seq,
            )
            question = (
                tokenizer.decode(input_ids[0]).replace("<pad>", "").replace("</s>", "")
            )
            predicted_sequence = predicted_sequence[0]
            gold_sequence = (
                tokenizer.decode(answer_ids).replace("<pad>", "").replace("</s>", "")
            )
            questions.append(question)
            if predicted_sequence[-1] == ",":
                predicted_sequence = predicted_sequence[:-1]
             
            predictions.append(predicted_sequence)
            targets.append(gold_sequence)
            if set(predicted_sequence.split(",")) == set(gold_sequence.split(",")):
                print(predicted_sequence)
                accurate += 1
            total += 1
        accuracy = accurate / total
        logger.log("accuracy is {} in epoch {}".format(accuracy, e))
        data = [(q, g, p) for q, g, p in zip(questions, targets, predictions)]
        with open("finetune_prediction_{}.json".format(args.num_train), "w") as f:
            json.dump(data, f)
                
        if accuracy > best_accuracy:
            logger.log(
                "accuracy improved from {} ----> {}".format(best_accuracy, accuracy)
            )
            
            torch.save(
                model.state_dict(), "{}_shot_finetuned.pt".format(args.num_train)
            )
            best_accuracy = accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_dir", type=str, default="toy.log")
    parser.add_argument("--model_name", type=str, default="t5-large")
    parser.add_argument("--prompt_dir", type=str, default="train_task_description.txt")
    parser.add_argument("--answer_dir", type=str, default="train_model_sequence.txt")

    parser.add_argument("--toy", action="store_true")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_seq", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--accumulate_steps", type=int, default=1)
    parser.add_argument("--warm_up_proportion", type=float, default=0.1)

    parser.add_argument("--num_train", type=int, default=10)

    args = parser.parse_args()

    set_seed(args)
    logger = Logger(args.logging_dir)

    train(args, logger)
