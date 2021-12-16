from typing import List, Optional
from dataclasses import dataclass
import pandas as pd
import json
from simple_ddl_parser import DDLParser


@dataclass
class Prompt:
    """
    :param db_create: List of all the CREATE TABLE statements in the database
    :param question: The question to turn into a SQL query or a query result
    """
    db_create: List[str]
    question: str

    def _srhink_table(self, table: str) -> str:
        result = DDLParser(table).run()
        if len(result) != 1:
            return False
        parsed_table = result[0]
        if 'table_name' not in parsed_table or 'columns' not in parsed_table:
            return False
        table = parsed_table['table_name'] + '(' + ','.join([x['name'] for x in parsed_table['columns']]) + ')'
        return table

    def _process_db_data(self):
        dbs = [self._srhink_table(x) for x in self.db_create]
        dbs = '\n'.join([x for x in dbs if x is not False])
        return dbs

    def to_text(self):
        text = self._process_db_data()
        text += '\n' + self.question
        return text

    def to_dict(self):
        return {'db_create': self._process_db_data(), 'question': self.question}

    def to_json(self):
        return json.dumps(self.to_dict())



@dataclass
class Response:
    """
    :param query: The query you should execute to get the desired result
    :param question: The desired result
    """
    query: str
    result: Optional[pd.DataFrame]

