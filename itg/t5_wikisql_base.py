from typing import Tuple, List
from transformers import AutoModelWithLMHead, AutoTokenizer
from controllers import Prompt
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from lightwood.helpers.torch import LightwoodAutocast
import numpy as np


class T5WSDataset(Dataset):
    def __init__(self, t5ws, data):
        super(T5WSDataset).__init__()
        self.t5ws = t5ws
        self.data = []
        for item in data:
            features, output = t5ws._prepare(item['prompt'], item['completion'])
            self.data.append({
                'input_ids': features['input_ids'],
                'attention_mask': features['attention_mask'],
                'labels': output
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Tuple[object, object]:
        return self.data[index]


class T5WS():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")
        self.model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-wikiSQL")

    def _prepare(self, prompt: Prompt, query: str = None):
        features = self.tokenizer([prompt.to_text()], return_tensors='pt')
        output = None
        if query is not None:
            output = self.tokenizer([output], return_tensors='pt')
        return features, output

    def __call__(self, prompt: Prompt) -> str:
        features, _ = self._prepare(prompt)
        output = self.model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'])
        return self.tokenizer.decode(output[0])

    def train(self, training_data: List[Tuple[Prompt, str]]):
        ds = T5WSDataset(self, training_data)
        dl = DataLoader(ds, batch_size=8, shuffle=True)

        parameters = self.model.parameters()
        optimizer = AdamW(parameters, lr=1e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(ds) * 100,
        )
        self.model = self.model.train()
        for epoch in range(100):
            total_loss = []
            for batch in dl:
                optimizer.zero_grad()

                with LightwoodAutocast():
                    predictions = self.model(input_ids=batch['input_ids'], 
                                             attention_mask=batch['attention_mask'], labels=batch['labels'])
                    loss = predictions[0]

                total_loss.append(loss.item())

                loss.backward()
                optimizer.step()
                scheduler.step()
                print(f'Current total loss: {np.mean(total_loss)} | Current epoch: {epoch}')
            print(f'\nTotal loss at end of epoch {epoch}: {np.mean(total_loss)} !\n')   