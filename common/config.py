# common/config.py

# These parameters must match on Client and Server.
# Based on the PyAPSI example, but typically you tune 'table_size' 
# based on how many tokens you expect in the database.
APSI_PARAMS = """
{
    "table_params": {
        "hash_func_count": 3,
        "table_size": 131072,         
        "max_items_per_bin": 92
    },
    "item_params": {"felts_per_item": 8},
    "query_params": {
        "ps_low_degree": 0,
        "query_powers": [1, 3, 4, 5, 8, 14, 20, 26, 32, 38, 41, 42, 43, 45, 46]
    },
    "seal_params": {
        "plain_modulus": 40961,
        "poly_modulus_degree": 4096,
        "coeff_modulus_bits": [40, 32, 32]
    }
}
"""

# Max length of the metadata label (the Record ID) in bytes
MAX_LABEL_LENGTH = 32
