from typing import Any, Iterable, List, Optional, Tuple, Union
import openai
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import pandas as pd
from itg.constant import OPEN_AI_API_KEY, MODEL_NAME


@dataclass_json
@dataclass
class Prompt:
    """
    :param db_create: List of all the CREATE TABLE statements in the database
    :param question: The question to turn into a SQL query or a query result
    """
    db_create: List[str]
    question: str


@dataclass
class Response:
    """
    :param query: The query you should execute to get the desired result
    :param question: The desired result
    """
    query: str
    result: Optional[pd.DataFrame]


class ITG:
    table_arr: List[str]

    def __init__(self) -> None:
        self.table_arr = []

    def __call__(self, question: str, database_connection: Any = None) -> Response:
        """
        :param nl_query: Natural language query you want turned into a SQL query
        :param database_connection: If provided, query the database and return the result directly
        """
        openai.api_key = OPEN_AI_API_KEY

        prompt = Prompt.from_dict({
            'db_create': self.table_arr,
            'question': question
        })
        
        str_prompt = prompt.to_json()
        # print(f'Requesting with prompt: {str_prompt}')

        openai_response = openai.Completion.create(
            model=MODEL_NAME,
            prompt=str_prompt,
            temperature=0,
            max_tokens=len(question) * 10,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"],
            n=1,
        )
        query = openai_response['choices'][0]['text']

        result = None
        if database_connection is not None:
            result = database_connection.execute(query)

        itg_response = Response(
            query=query,
            result=result
        )
        return itg_response

    def register(self, db_tables: Union[List[str], List[Tuple[pd.DataFrame, str]], Tuple[pd.DataFrame, str]]) -> None:
        """
        :param db_tables: A list of the CREATE statements for all the queries in the database
        """
        if not isinstance(db_tables, list):
            db_tables = [db_tables]

        for table in db_tables:
            if isinstance(table, tuple):
                table = self.dataframe_to_create(table[0], table[1])

            self.table_arr.append(table)

    def dataframe_to_create(self, df: pd.DataFrame, name: str) -> str:
        """
        :param name: The name of the table represented by the dataframe (if unsure just put the name of the variable this df is assigned to)

        :return: A create statement for a table analogous to your dataframe
        """ # noqa
        create_query = f'CREATE TABLE {name} ('
        for col in df.columns:
            sql_type = str(df[col].dtype)
            if 'object' in sql_type:
                sql_type = 'TEXT'
            elif sql_type.startswith('f'):
                sql_type = 'FLOAT'
            elif sql_type.startswith('i'):
                sql_type = 'INT'
            elif 'datetime' in sql_type:
                sql_type = 'DATETIME'
            else:
                sql_type = 'DATE'
            create_query += f'\n{col} {sql_type}'
        create_query += '\n)'

        return create_query


