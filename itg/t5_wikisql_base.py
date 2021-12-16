from copy import deepcopy
from typing import Dict
from transformers import AutoModelWithLMHead, AutoTokenizer
from itg.types import Prompt, TrainData
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from lightwood.helpers.torch import LightwoodAutocast
import numpy as np
import torch
import random
from lightwood.helpers.device import get_devices


class T5WSDataset(Dataset):
    def __init__(self, t5ws: object, data: TrainData):
        super(T5WSDataset).__init__()
        features = t5ws.tokenizer([x['prompt'].to_text() for x in data], return_tensors='pt', truncation=True,
                                  padding=True)
        outputs = t5ws.tokenizer([x['completion'] for x in data], return_tensors='pt', truncation=True,
                                 padding=True)
        self.decoder_attention_mask = outputs['attention_mask']
        outputs = outputs['input_ids']
        outputs = torch.tensor(outputs)
        self.features = features
        self.outputs = outputs

    def __len__(self):
        return len(self.features['input_ids'])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        batch_sample = {
            'input_ids': self.features['input_ids'][index],
            'attention_mask': self.features['attention_mask'][index],
            'labels': self.outputs[index],
            # 'decoder_input_ids': self.outputs[index],  # Still not sure this works or helps
            'decoder_attention_mask': self.decoder_attention_mask[index]  # Still not sure this works or helps
        }
        return batch_sample


class T5WS():
    def __init__(self, save_path: str = None):
        self.device = get_devices()[0]
        if save_path is None:
            print('Save path not provided, initializing untrained model')
            self.tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")
            self.model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")
        else:
            print(f'Initializing model and tokenizer from: {save_path}')
            self.tokenizer = AutoTokenizer.from_pretrained(save_path)
            self.model = AutoModelWithLMHead.from_pretrained(save_path)

        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        self.best_accuracy = -pow(2, 63)

    def save(self, epoch: int):
        self.best_model.save_pretrained(f'weights/t5-base-mindsdb-sqlizer-{epoch}')
        self.tokenizer.save_pretrained(f'weights/t5-base-mindsdb-sqlizer-{epoch}')

    def __call__(self, prompt: Prompt) -> str:
        features = self.tokenizer([prompt.to_text()], return_tensors='pt', truncation=True, padding=True)
        output = self.model.generate(input_ids=features['input_ids'].to(self.device),
                                     attention_mask=features['attention_mask'] .to(self.device))
        return self.tokenizer.decode(output[0])

    def train(self, training_data: TrainData):
        random.seed(4372373)
        random.shuffle(training_data)
        nr_epochs = 200
        batch_size = 8
        self.best_model = deepcopy(self.model.cpu())

        ds_train = T5WSDataset(self, training_data[:int(len(training_data) * 0.8)])
        print(f'Train data length: {len(ds_train)}')
        dlt = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        ds_eval = training_data[int(len(training_data) * 0.8):]
        parameters = self.model.parameters()
        optimizer = AdamW(parameters, lr=1e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=5,
            num_training_steps=len(ds_train) * nr_epochs,
        )

        for epoch in range(nr_epochs):
            total_loss = []
            self.model = self.model.to(self.device)
            self.model = self.model.train()
            step = 0
            for batch in dlt:
                step += 1
                optimizer.zero_grad()

                for k in batch:
                    batch[k] = batch[k].to(self.device)
                with LightwoodAutocast():
                    predictions = self.model(**batch)
                    loss = predictions[0]

                nl = loss.item()
                if 'nan' in str(nl).lower():
                    print('Got a loss equal to nan, skipping this one!')
                    optimizer.zero_grad()
                    continue
                total_loss.append(nl)
                loss.backward()
                optimizer.step()
                scheduler.step()

                print(f'Current total loss: {np.mean(total_loss)} | Current epoch: {epoch} [Step {step} - \
{round(100 * (batch_size * step) / len(ds_train), 2)}% done]')
            print(f'\nTotal loss at end of epoch {epoch}: {np.mean(total_loss)} !\n')
            eval_acc = self.evaluate(ds_eval)
            if eval_acc > self.best_accuracy:
                print(f'New best model with evaluation accuracy of: {eval_acc}!')
                self.best_model = deepcopy(self.model.cpu())
                self.save(epoch)

        self.model = self.best_model.to(self.device)

    def evaluate(self, ds_eval: TrainData) -> float:
        self.model = self.model.eval()
        correct = 0
        total = len(ds_eval)
        for item in ds_eval:
            with torch.no_grad():
                with LightwoodAutocast():
                    predicted_completion = self(item['prompt']).replace(
                        '</s>', '').replace('<pad>', '').lstrip(' ').rstrip(' ')
                    real_completion = item['completion']

                    if predicted_completion.lower() == real_completion.lower():
                        correct += 1
                    if random.randint(0, 50) == 5:
                        print(f'[Illustrative Sample]\nInput: {item["prompt"].to_text()}')
                        print(f'Predicted: "{predicted_completion}"\nReal: "{real_completion}"')
        print(f'\n\nModel was correct for {correct} queries ({round(100 * correct / total, 2)}%)\n\n')
        return correct / total
