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


# Try to only train on and take into consideration simple queries for now
def is_simple_query(query):
    complicated = False

    lq = query.lower()
    if ' join ' in lq:
        complicated = True
    if len(lq) > 100:
        complicated = True
    if ' order ' in lq:
        complicated = True
    if ' limit ' in lq:
        complicated = True

    if not complicated:
        return True
    else:
        return False


def sparc_to_prompt() -> TrainData:
    raw_data = json.load(open('sparc/train.json', 'r'))
    raw_data += json.load(open('sparc/dev.json', 'r'))
    db_cache = {}

    data = []
    for example in raw_data:
        if example['database_id'] not in db_cache:
            db_cache[example['database_id']] = parse_db_file(example['database_id'])
        db_data = db_cache[example['database_id']]
        if len(db_data) == 0:
            continue

        for interaction in example['interaction']:
            real_query = interaction['query']
            if not is_simple_query(real_query):
                continue
            question = interaction['utterance']
            prompt = Prompt(db_create=db_data, question=question)
            stringified_prompt = prompt.to_json()
            if len(stringified_prompt) > MAX_TRAIN_LENGTH:
                # Most of the data is in the table, so if size is exceded by more than 100, assume all are invalid
                if len(stringified_prompt) > MAX_TRAIN_LENGTH + 100:
                    break
                continue

            data.append({'prompt': stringified_prompt, 'completion': real_query})

            if len(data) > TRAIN_ON:
                return data
    return data


def spider_to_prompt():
    raw_data = json.load(open('spider/train_spider.json', 'r'))
    raw_data += json.load(open('spider/dev.json', 'r'))
    db_cache = {}

    data = []
    for interaction in raw_data:
        if interaction['db_id'] not in db_cache:
            db_cache[interaction['db_id']] = parse_db_file(interaction['db_id'])
        db_data = db_cache[interaction['db_id']]
        if len(db_data) == 0:
            continue

        real_query = interaction['query']
        if not is_simple_query(real_query):
            continue
        question = interaction['question']
        prompt = Prompt(db_create=db_data, question=question)
        stringified_prompt = prompt.to_json()
        if len(stringified_prompt) > MAX_TRAIN_LENGTH:
            continue

        data.append({'prompt': prompt, 'completion': real_query})

        if len(data) > TRAIN_ON:
            return data

    return data


def train():
    training_data = sparc_to_prompt()
    print(f'Train data length: {len(training_data)}')

    # @TODO: Figure out how to switch to the python API
    for config_options in [
        ('json', 1),
        ('text', 1),
        ('json', 5),
        ('text', 5),
    ]:
        prompt_fmt = config_options[0]
        n_epochs = config_options[1]

        if prompt_fmt == 'json':
            stringified_data = [{'prompt': x['prompt'].to_json(), 'completion': x['completion']} for x in training_data]
        elif prompt_fmt == 'text':
            stringified_data = [{'prompt': x['prompt'].to_text(), 'completion': x['completion']} for x in training_data]

        train_file = f'train_file_{prompt_fmt}_{n_epochs}'
        with open(train_file, 'w') as fp:
            fp.write('\n'.join(stringified_data))

        training_statements = [
            f'export OPENAI_API_KEY="{OPEN_AI_API_KEY}"',
            f'openai api fine_tunes.create -t {train_file} -m {BASE_MODEL} --n_epochs {n_epochs}'
        ]
        os.system(' && '.join(training_statements))


def test():
    testing_data = spider_to_prompt()
    print(f'Test data length: {len(testing_data)}')

    for model_name in [
        '???'
    ]:
        nr_correct = 0
        nr_incorrect = 0
        for prompt, real_query in testing_data:
            itg = ITG(model_name)
            itg.register(prompt.db_create)
            response = itg(prompt.question)
            predicted_query = response.result
            correct = predicted_query == real_query
            print(f"""
Predicted query: {predicted_query}
Real query: {real_query}
Correct: {correct}
            """)
            if correct:
                nr_correct += 1
            else:
                nr_incorrect += 1

    print(f'Number of incorrect observations: {nr_incorrect}')
    print(f'Number of correct observations: {nr_correct}')


if __name__ == '__main__':
    train()
    test()
