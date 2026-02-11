GRAPH = {
    "A": {
        "B": 2
    },
    "B": {
        "C": 2,
        "E": 1
    },
    "C": {
        "D": 2,
        "H": 1
    },
    "D": {},
    "H": {
        "I": 9,
        "F": 1
    },
    "I": {
        "B": 1,
        "K": 2
    },
    "K": {},
    "E": {
        "F": 6,
        "I": 1
    },
    "F": {
        "E": 2,
        "C": 1
    },
    "G": {
        "H": 2
    }
}

TURN_TABLE = {
    ('A', 'B'): 'STRAIGHT',
    ('B', 'C'): 'RIGHT',
    ('B', 'E'): 'STRAIGHT',
    ('C', 'H'): 'STRAIGHT',
    ('C', 'D'): 'RIGHT',
    ('H', 'I'): 'STRAIGHT',
    ('H', 'F'): 'SEMI-LEFT',
    ('E', 'F'): 'STRAIGHT',
    ('E', 'I'): 'LEFT',
    ('F', 'C'): 'STRAIGHT',
    ('F', 'E'): 'RIGHT',
    ('I', 'K'): 'STRAIGHT',
    ('I', 'B'): 'SEMI-LEFT',
    ('G', 'H'): 'STRAIGHT',
}

RFID_MAP = {
    "E59CA94090": "A",
    "2501E54687": "B",
    "8516A34070": "C",
    "A5467440D7": "D",
    "95BB8C40E2": "E",
    "C57C6B4092": "F",
    "757BB640F8": "G",
    "D25CD6055D": "H",
    "2AE5D6051C": "I",
    "8B23D6057B": "K"
}

TURN_CONFIG = {
    "90_DEG": 0.8,
    "45_DEG": 0.5
}
