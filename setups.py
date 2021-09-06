SETUPS = {
    'empty': {
        'size': 41,
        'blocked': tuple(),
        'target_neuron': (40, 40),
        't_max': 500,
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
            (slice(10, 30), slice(10, 30)),
            ),
        'target_neuron': (40, 40),
        't_max': 5000,
        'randomize_neurons': False,
        'thalamic_input': False,
        },
}
