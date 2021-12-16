from re import M
from typing import Tuple, List
from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from controllers import Prompt
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from lightwood.helpers.torch import LightwoodAutocast
import numpy as np
import torch
import random


class T5WSDataset(Dataset):
    def __init__(self, t5ws, data):
        super(T5WSDataset).__init__()

        features = t5ws.tokenizer([x['prompt'].to_text() for x in data], return_tensors='pt', truncation=True, padding=True)
        outputs = t5ws.tokenizer([x['completion'] for x in data], return_tensors='pt', truncation=True, padding=True)
        outputs = outputs['input_ids']
        outputs = [[(label if label != t5ws.tokenizer.pad_token_id else -100)
                    for label in labels_example] for labels_example in outputs]
        outputs = torch.tensor(outputs)
        self.features = features
        self.outputs = outputs
        
    def __len__(self):
        return len(self.features['input_ids'])

    def __getitem__(self, index) -> Tuple[object, object]:
        batch_sample = {
            'input_ids': self.features['input_ids'][index],
            'attention_mask': self.features['attention_mask'][index],
            'labels': self.outputs[index]
        }
        return batch_sample


class T5WS():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")
        self.model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL").cuda()
        # self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        # self.model = T5ForConditionalGeneration.from_pretrained("t5-small").cuda()

    def __call__(self, prompt: Prompt) -> str:
        features = self.tokenizer([prompt.to_text()], return_tensors='pt', truncation=True, padding=True)
        output = self.model.generate(input_ids=features['input_ids'].cuda(),
                                     attention_mask=features['attention_mask'].cuda())
        return self.tokenizer.decode(output[0])

    def train(self, training_data: List[Tuple[Prompt, str]]):
        random.seed(14212)
        random.shuffle(training_data)
        nr_epochs = 20
        batch_size = 16

        dst = T5WSDataset(self, training_data[:int(len(training_data) * 0.8)])
        print(f'Train data length: {len(dst)}')
        dlt = DataLoader(dst, batch_size=batch_size, shuffle=True)
        rawv = training_data[int(len(training_data) * 0.8):]

        parameters = self.model.parameters()
        optimizer = AdamW(parameters, lr=1e-4)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=30,
            num_training_steps=len(dst) * nr_epochs,
        )
        
        for epoch in range(nr_epochs):
            total_loss = []
            self.model = self.model.train()
            step = 0
            for batch in dlt:
                step += 1
                optimizer.zero_grad()

                for k in batch:
                    batch[k] = batch[k].cuda()
                with LightwoodAutocast():
                    predictions = self.model(**batch)
                    loss = predictions[0]
                total_loss.append(loss.item())

                loss.backward()
                optimizer.step()
                scheduler.step()
                print(f'Current total loss: {np.mean(total_loss)} | Current epoch: {epoch} [Step {step} - \
{round(100 * (batch_size * step) / len(dst), 2)}% done]')
            print(f'\nTotal loss at end of epoch {epoch}: {np.mean(total_loss)} !\n')

            self.model = self.model.eval()
            correct = 0
            total = len(rawv)
            for item in rawv:
                with torch.no_grad():
                    with LightwoodAutocast():
                        predicted_completion = self(item['prompt']).lstrip(' ').rstrip(' ').replace('</s>','').replace('<pad>', '')
                        real_completion = item['completion']

                        if predicted_completion.lower() == real_completion.lower():
                            correct += 1
                        if random.randint(0, 50) == 5:
                            print(f'[Illustrative Sample] Predicted: {predicted_completion}\nReal: {real_completion}')
            print(f'\n\nModel was correct for {correct} queries ({100 * correct / total}%)\n\n')