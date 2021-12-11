from typing import Dict
from controllers import ITG, Prompt
import json
import os
from typing import List
from itg.constant import OPEN_AI_API_KEY, BASE_MODEL, MAX_TRAIN_LENGTH, TRAIN_ON


itg = ITG()
TrainData = List[Dict[Prompt, str]]


def parse_db_file(name: str) -> str:
    # Names of the sql files seem to vary for no(?) reason
    sql_file_name = None
    for filename in os.listdir(f'sparc/database/{name}/'):
        if filename.endswith('.sql'):
            sql_file_name = filename
            break

    if sql_file_name is None:
        print(f'No sql data found for database: {name}')
        return []

    with open(f'sparc/database/{name}/{sql_file_name}', 'r') as fp:
        fp.readline()
        sql = fp.read().replace('"', '')
    tables = sql.split(';')
    tables = [x for x in tables if 'INSERT' not in x]
    return tables


def sparc_to_prompt() -> TrainData:
    raw_train_data = json.load(open('sparc/train.json', 'r'))

    train_data = []
    for example in raw_train_data:
        db_data = parse_db_file(example['database_id'])
        if len(db_data) == 0:
            continue

        for interaction in example['interaction']:
            real_query = interaction['query']
            question = interaction['utterance']
            prompt = Prompt(db_create=db_data, question=question)
            stringified_prompt = prompt.to_json()
            if len(stringified_prompt) > MAX_TRAIN_LENGTH:
                print(f'Too big at length: {len(stringified_prompt)}')
                # Most of the data is in the table, so if size is exceded by more than 100, assume all are invalid
                if len(stringified_prompt) > MAX_TRAIN_LENGTH + 100:
                    print(f'Skipping example with database: {example["database_id"]}')
                    break
                continue

            print(f'Not too big at length: {len(stringified_prompt)}')

            train_data.append({'prompt': stringified_prompt, 'completion': real_query})

            if len(train_data) > TRAIN_ON:
                return train_data


def train(training_data: TrainData, name: str):
    # @TODO: Figure out how to switch to the python API
    with open('train_file', 'w') as fp:
        stringified_data = [json.dumps(x) for x in training_data]
        fp.write('\n'.join(stringified_data))
    training_statements = [
        f'export OPENAI_API_KEY="{OPEN_AI_API_KEY}"',
        f'openai api fine_tunes.create -t train_file -m {name} --n_epochs 10'
    ]
    os.system(' && '.join(training_statements))


def main():
    train_data = sparc_to_prompt()
    train(train_data, BASE_MODEL)


if __name__ == '__main__':
    main()
