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

for i in range(1):
    print('\n\n\n\n')
    try:
        ask = 'How much money did we make for each product?'
        result1 = itg(ask)
        print(f'We asked: {ask}\nGot the query: {result1.query}')

        ask = 'How much money did each store earn every day?'
        result2 = itg(ask)
        print(f'We asked: {ask}\nGot the query: {result2.query}')

        ask = 'How many items did each shop sell?'
        result3 = itg(ask)
        print(f'We asked: {ask}\nGot the query: {result3.query}')

        query_result = sql_query(result1.query, sales=sales)
        print(f'Using the first query we got: {query_result}')

        query_result = sql_query(result2.query, sales=sales)
        print(f'Using the second query we got: {query_result}')

        query_result = sql_query(result3.query, sales=sales)
        print(f'Using the third query we got: {query_result}')

        print(f'Got all queries to execute after {i+1} attempts!')
        break
    except Exception as e:
        print(f'Iteration failed with error: {e}')
