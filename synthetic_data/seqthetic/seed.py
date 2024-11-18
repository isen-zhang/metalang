import numpy as np


def make_seed():
    return np.random.SeedSequence().entropy


def spawn_rng_list(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    new_rngs = [np.random.Generator(rng.bit_generator.spawn(1)[0]) for _ in range(n)]
    return new_rngs


def get_rngs(seed: int, spec: list[str | tuple[str, int]]):
    ss = np.random.SeedSequence(seed)
    all_seed_seqs = ss.spawn(len(spec))
    all_rngs: list[np.random.Generator | list[np.random.Generator]] = []
    for s, spec_item in zip(all_seed_seqs, spec):
        if isinstance(spec_item, tuple):
            _, count = spec_item
            rngs = [np.random.default_rng(s) for s in ss.spawn(count)]
            all_rngs.append(rngs)
        else:
            rng = np.random.default_rng(s)
            all_rngs.append(rng)
    return all_rngs

if __name__ == '__main__':
    rng = np.random.default_rng()
    spawn_rng_list(rng, 4)
