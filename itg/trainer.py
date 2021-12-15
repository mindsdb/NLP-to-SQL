from typing import Dict
from controllers import ITG, Prompt
import json
import os
from typing import List
from itg.constant import OPEN_AI_API_KEY, BASE_MODEL, MAX_TRAIN_LENGTH, TRAIN_ON
from t5_wikisql_base import T5WS


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

            data.append({'prompt': prompt, 'completion': real_query})

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


def train_openai():
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
            fp.write('\n'.join([json.dumps(x) for x in stringified_data]))

        training_statements = [
            f'export OPENAI_API_KEY="{OPEN_AI_API_KEY}"',
            f'openai api fine_tunes.create -t {train_file} -m {BASE_MODEL} --n_epochs {n_epochs}'
        ]
        os.system(' && '.join(training_statements))


def test_openai():
    testing_data = spider_to_prompt()
    print(f'Test data length: {len(testing_data)}')

    result = os.popen(f'export OPENAI_API_KEY="{OPEN_AI_API_KEY}" && openai api fine_tunes.list').read()
    result = json.loads(result)['data']
    models = [(x['fine_tuned_model'], x['training_files'][0]['filename']) for x in result
              if x['fine_tuned_model'] is not None
              and ('2021-12-13' in x['fine_tuned_model']
              or '2021-12-14' in x['fine_tuned_model'])]

    for model_name, fmt in models:
        fmt = 'text' if '_text_' in fmt else fmt
        fmt = 'json' if '_json_' in fmt else fmt
        nr_correct = 0
        nr_incorrect = 0
        for item in testing_data[0:50]:
            prompt = item['prompt']
            real_query = item['completion']
            itg = ITG(model_name, fmt)
            itg.register(prompt.db_create)
            response = itg(prompt.question)
            predicted_query = response.query
            correct = predicted_query.lower() == real_query.lower()

            '''
            print(f"""
Predicted query: {predicted_query}
Real query: {real_query}
Correct: {correct}
            """)
            '''

            if correct:
                nr_correct += 1
            else:
                nr_incorrect += 1

        print(f'Results for model: {model_name}')
        print(f'Number of incorrect observations: {nr_incorrect}')
        print(f'Number of correct observations: {nr_correct}')


def train_t5ws():
    training_data = sparc_to_prompt()
    print(f'Test data length: {len(training_data)}')
    model = T5WS()
    model.train(training_data)

if __name__ == '__main__':
    # train_openai()
    # test_openai()
    train_t5ws()
