
These two scripts generate the data preprocesses.

1. Generate Edge: 
saves:
- id_mappings: maps the account_id to the new account ids. including of the non-numerical accounts
- edge tensor which is the edge_index shape [288785789, 2]
- edge features:  288785789, 9
    features ['weekday', 'channel', 'payment_system', 'category_0', 'category_1', 'category_2', 'amount', 'seconds_of_day', 'days_of_year']

- label_encoding: maps every column to its original values


2. generate nodes
- save label_encoders node for ["assigned_bank_type", "assigned_bank"]
- node features [1813407, 4] (order according to id_mapped)
- features are ["age", "initial_balance", "assigned_bank_type", "assigned_bank"]