# Constants related to the BACH dataset (https://iciar2018-challenge.grand-challenge.org/Dataset/)
CLASS_DICT_BACH = {'Normal': 0, 'Benign': 1, 'InSitu': 2, 'Invasive': 3}
PHASE_DICT_BACH = {
    'train': {'start': 0.0, 'stop': 1.0},
    'valid': {'start': 0.75, 'stop': 1.0},
    'test': {'start': 0.0, 'stop': 1.0}
}

# Constants related to the BRACS dataset (https://www.bracs.icar.cnr.it/)
CLASS_DICT_BRACS_TROI_LATEST = {'0_N': 0, '1_PB': 1, '2_UDH': 2, '3_FEA': 3, '4_ADH': 4, '5_DCIS': 5, '6_IC': 6}
CLASS_DICT_BRACS_TROI_PREVIOUS = {'0_N': 0, '1_PB': 1, '2_UDH': 2, '3_ADH': 3, '4_FEA': 4, '5_DCIS': 5, '6_IC': 6}
CLASS_DICT_BRACS_WSI = {'Type_N': 0, 'Type_PB': 1, 'Type_UDH': 2, 'Type_FEA': 3, 'Type_ADH': 4, 'Type_DCIS': 5,
                        'Type_IC': 6}

PHASE_DICT_BRACS_TROI = {
    'train': {'start': 0.0, 'stop': 1.},
    'valid': {'start': 0.0, 'stop': 1.},
    'test': {'start': 0.0, 'stop': 1.}
}
