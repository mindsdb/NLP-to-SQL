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
        self.t5ws = t5ws
        self.data = []
        for item in data:
            features, output = t5ws._prepare(item['prompt'], item['completion'])
            self.data.append({
                'input_ids': features['input_ids'][0, :].cuda(),
                'attention_mask': features['attention_mask'][0, :].cuda()
            })
            if output is not None:
                self.data[-1]['labels'] = output[0, :].cuda()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Tuple[object, object]:
        return self.data[index]


class T5WS():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")
        self.model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL").cuda()
        # self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        # self.model = T5ForConditionalGeneration.from_pretrained("t5-small").cuda()
        
    def _prepare(self, prompt: Prompt, query: str = None):
        features = self.tokenizer([prompt.to_text()], return_tensors='pt',
                                  truncation=True, padding='max_length', max_length=512)
        output = None
        if query is not None:
            output = self.tokenizer([query], return_tensors='pt', truncation=True, padding='max_length', max_length=512)
            output = output['input_ids']
            output = [[(label if label != self.tokenizer.pad_token_id else -100)
                       for label in labels_example] for labels_example in output]
            output = torch.tensor(output)
        return features, output

    def __call__(self, prompt: Prompt) -> str:
        features, _ = self._prepare(prompt)
        output = self.model.generate(input_ids=features['input_ids'].cuda(),
                                     attention_mask=features['attention_mask'].cuda())
        return self.tokenizer.decode(output[0])

    def train(self, training_data: List[Tuple[Prompt, str]]):
        random.seed(14212)
        random.shuffle(training_data)
        nr_epochs = 20
        batch_size = 4

        dst = T5WSDataset(self, training_data[:int(len(training_data) * 0.8)])
        dlt = DataLoader(dst, batch_size=batch_size, shuffle=True)
        rawv = training_data[int(len(training_data) * 0.8):]

        parameters = self.model.parameters()
        optimizer = AdamW(parameters, lr=1e-5)
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

                with LightwoodAutocast():
                    predictions = self.model(**batch)
                    loss = predictions[0]

                total_loss.append(loss.item())

                loss.backward()
                optimizer.step()
                scheduler.step()
                print(f'Current total loss: {np.mean(total_loss)} | Current epoch: {epoch} [Step {step},\
                    {100 * len(dst)/ (batch_size * step)}% done]')
            print(f'\nTotal loss at end of epoch {epoch}: {np.mean(total_loss)} !\n')

            self.model = self.model.eval()
            correct = 0
            total = len(rawv)
            for item in rawv:
                with torch.no_grad():
                    with LightwoodAutocast():
                        predicted_completion = self(item['prompt'])
                        real_completion = item['completion']

                        if predicted_completion.lower() == real_completion.lower():
                            correct += 1
                        print(f'Predicted: {predicted_completion}\nReal: {real_completion}\n')
            print(f'\n\nModel was correct for {correct} queries ({100 * correct / total}%)\n\n')