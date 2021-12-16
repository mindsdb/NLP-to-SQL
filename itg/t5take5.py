from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
from trainer import sparc_to_prompt


class T5T5():
    def __init__(self):
        pass

    def compute_metrics(self, eval_preds):
        print('Called compute metric !!')
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        total = len(decoded_preds)
        correct = 0
        for i in range(len(decoded_preds)):
            if decoded_preds[i].lower() == decoded_labels[i].lower():
                correct += 1
        return {'p_correct': correct / total}

    def train(self, data):
        data = data[0:100]
        print('Initializing models')
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        print('Creating datasets')

        input_dict = self.tokenizer([x['prompt'].to_text() for x in data], return_tensors='pt', truncation=True, padding=True)
        print(input_dict.keys())
        output_dcit = self.tokenizer([x['completion'] for x in data], return_tensors='pt', truncation=True, padding=True)
        input_dict['decoder_input_ids'] = output_dcit['input_ids']
        input_dict['decoder_attention_mask'] = output_dcit['attention_mask']
        dataset = Dataset.from_dict(input_dict)

        print('Creating and starting trainer')
        # Initialize our Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            train_dataset=dataset,
            eval_dataset=dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            args=Seq2SeqTrainingArguments(
                output_dir='hft',
                overwrite_output_dir=True,
                do_train=True,
                do_eval=True,
                num_train_epochs=20,
                generation_max_length=512,
            )
        )

        ev = trainer.evaluate()
        print(ev)
        trainer.train()
        trainer.evaluate()
        #trainer.save_model()
        #metrics = train_result.metrics
        #trainer.log_metrics("train", metrics)
        #trainer.save_metrics("train", metrics)
        #trainer.save_state()


if __name__ == '__main__':
    t5t5 = T5T5()
    t5t5.train(sparc_to_prompt())