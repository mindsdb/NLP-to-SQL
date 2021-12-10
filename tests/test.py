import pandas as pd
from itg import itg
from dfsql import sql_query


sales = pd.DataFrame({
    'product': ['toothbrush', 'ribeye', 'cologne', 'ribeye', 'parrot', 'cage'],
    'store': ['Tesco Highstreet', 'Tesco Downtown', 'Tesco Downtown', 'Tesco Downtown',
              'Tesco Highstreet', 'Tesco Highstreet'],
    'price': [44, 51, 11, 51, 22, 11],
    'datetime': ['2021-10-01 00:00:00', '2021-10-01 05:44:00', '2021-10-01 23:00:01', '2021-10-02 00:43:01',
                 '2021-10-02 22:04:01', '2021-10-02 23:11:50']
})

itg.register((sales, 'sales'))

for i in range(10):
    print('\n\n\n\n')
    try:
        ask = 'How much money did we make for each product?'
        result = itg(ask)
        print(f'We asked: {ask}\nGot the query: {result.query}')
        query_result = sql_query(result.query, sales=sales)
        print(f'Using the query we got: {query_result}')

        ask = 'How many items did each shop sell?'
        result = itg(ask)
        print(f'We asked: {ask}\nGot the query: {result.query}')
        query_result = sql_query(result.query, sales=sales)
        print(f'Using the query we got: {query_result}')

        ask = 'How much money did each store earn every day?'
        result = itg(ask)
        print(f'We asked: {ask}\nGot the query: {result.query}')
        query_result = sql_query(result.query, sales=sales)
        print(f'Using the query we got: {query_result}')

        print(f'Got all queries to execute after {i+1} attempts!')
        break
    except Exception as e:
        print(f'Iteration failed with error: {e}')
        pass
