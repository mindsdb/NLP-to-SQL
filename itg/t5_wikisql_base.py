from typing import Tuple, List
from transformers import AutoModelWithLMHead, AutoTokenizer
from controllers import Prompt


class T5WS()
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
        for prompt, query in training_data:
            features, output = self._prepare(prompt, query)
