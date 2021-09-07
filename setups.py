SETUPS = {
    'empty': {
        'size': 101,
        'blocked': tuple(),
        'target_neuron': (50, 50),
        't_max': 100,
        'randomize_neurons': False,
        'thalamic_input': False,
        },
    's_maze': {
        'size': 41,
        'blocked': (
            (slice(25, 32), slice(10, None)),
            (slice(10, 15), slice(None, 32)),
            ),
        'target_neuron': (40, 40),
        't_max': 1300,
        'randomize_neurons': False,
        'thalamic_input': False,
        },
    'central_block': {
        'size': 41,
        'blocked': (
            (slice(8, 30), slice(8, 30)),
            ),
        'target_neuron': (40, 40),
        't_max': 2750,
        'randomize_neurons': True,
        'thalamic_input': False,
        },
    'complex_maze': {
        'size': 60,
        'blocked': (
            (slice(0, 2), slice(56, None)),
            (slice(6, 8), slice(None, 8)),
            (slice(6, 8), slice(14, 24)),
            (slice(6, 8), slice(32, 52)),
            (slice(8, 10), slice(6, 8)),
            (slice(8, 10), slice(14, 24)),
            (slice(8, 10), slice(32, 34)),
            (slice(10, 12), slice(14, 24)),
            (slice(10, 12), slice(32, 34)),
            (slice(12, 14), slice(14, 24)),
            (slice(12, 14), slice(32, 34)),
            (slice(14, 16), slice(14, 24)),
            (slice(14, 16), slice(32, 52)),
            (slice(16, 18), slice(6, 8)),
            (slice(16, 18), slice(50, 52)),
            (slice(18, 20), slice(6, 8)),
            (slice(18, 20), slice(50, 52)),
            (slice(20, 22), slice(6, 8)),
            (slice(20, 22), slice(50, 52)),
            (slice(22, 24), slice(6, 8)),
            (slice(22, 24), slice(34, 36)),
            (slice(22, 24), slice(50, 52)),
            (slice(24, 26), slice(6, 24)),
            (slice(24, 26), slice(32, 40)),
            (slice(24, 26), slice(50, 52)),
            (slice(26, 28), slice(6, 26)),
            (slice(26, 28), slice(32, 34)),
            (slice(28, 30), slice(32, 34)),
            (slice(30, 32), slice(32, 34)),
            (slice(30, 32), slice(58, 60)),
            (slice(32, 34), slice(None, 4)),
            (slice(32, 34), slice(32, 34)),
            (slice(32, 34), slice(32, 50)),
            (slice(32, 34), slice(58, 60)),
            (slice(34, 36), slice(None, 4)),
            (slice(34, 36), slice(12, 34)),
            (slice(34, 36), slice(32, 50)),
            (slice(34, 36), slice(58, 60)),
            (slice(36, 38), slice(None, 4)),
            (slice(36, 38), slice(12, 20)),
            (slice(36, 38), slice(26, 34)),
            (slice(38, 40), slice(None, 4)),
            (slice(38, 40), slice(12, 20)),
            (slice(38, 40), slice(26, 34)),
            (slice(40, 42), slice(None, 4)),
            (slice(40, 42), slice(26, 34)),
            (slice(42, 44), slice(26, 34)),
            (slice(42, 44), slice(40, None)),
            (slice(44, 46), slice(26, 34)),
            (slice(46, 48), slice(8, 34)),
            (slice(48, 50), slice(8, 34)),
            (slice(50, 52), slice(8, 34)),
            (slice(50, 52), slice(42, 44)),
            (slice(50, 52), slice(56, None)),
            (slice(52, 54), slice(8, 34)),
            (slice(52, 54), slice(42, 44)),
            (slice(54, 56), slice(40, 42)),
            (slice(54, 56), slice(42, 44)),
            (slice(56, 58), slice(40, 44)),
            (slice(56, 58), slice(42, 44)),
            (slice(56, 58), slice(50, 52)),
            (slice(58, 60), slice(2, 4)),
            (slice(58, 60), slice(40, 42)),
            (slice(58, 60), slice(42, 44)),
            (slice(58, 60), slice(50, 52))
        ),
        'target_neuron': (58, 58),
        't_max': 2500,
        'randomize_neurons': False,
        'thalamic_input': False,
        },
}
