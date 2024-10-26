from mlfz.functional import vanilla as f


def test_output_type():
    for func in [
        f.log_prime,
        f.relu,
        f.relu_prime,
        f.sigmoid,
        f.sigmoid_prime,
        f.tanh,
        f.tanh_prime,
    ]:
        func(1.0)
