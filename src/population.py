import numpy as np


def _beta_centered(rng, n, center, concentration=20.0):
    """
    Genera n valores en [0,1] a partir de una Beta cuya media es 'center'.
    concentration controla qué tan apretada está la distribución alrededor
    del centro: valores mas altos -> menos dispersión.
    """
    if center <= 0:
        return np.zeros(n)
    a = center * concentration
    b = (1 - center) * concentration
    return rng.beta(a, b, size=n)


def generate_population(N, delta_center, kappa_center, seed,
                         W_min=10_000, W_max=500_000,
                         n_min=5, n_max=30):
    """
    Genera una población de N traders para un escenario dado.

    delta_center, kappa_center: valores centrales que vienen de la tabla
        de escenarios (ej. 0, 0.3, 0.8).
    seed: semilla maestra única para todo el escenario (reproducibilidad).

    Regresa un dict con 4 arreglos: delta, kappa, W, n -- uno por trader.
    """
    # Una seed maestra -> 4 sub-semillas independientes, una por variable.
    seed_seq = np.random.SeedSequence(seed)
    seed_delta, seed_kappa, seed_W, seed_n = seed_seq.spawn(4)

    rng_delta = np.random.default_rng(seed_delta)
    rng_kappa = np.random.default_rng(seed_kappa)
    rng_W = np.random.default_rng(seed_W)
    rng_n = np.random.default_rng(seed_n)

    delta = _beta_centered(rng_delta, N, delta_center)
    kappa = _beta_centered(rng_kappa, N, kappa_center)

    # W y n no dependen de delta/kappa (Camino A: todo independiente)
    W = rng_W.uniform(W_min, W_max, size=N)
    n = rng_n.integers(n_min, n_max + 1, size=N)

    return {"delta": delta, "kappa": kappa, "W": W, "n": n}


if __name__ == "__main__":
    # Ejemplo: escenario 3 de la tabla ("Disposition only, high": delta=0.8, kappa=0)
    pop = generate_population(N=1000, delta_center=0.8, kappa_center=0.0, seed=12345)

    print("delta -> media:", pop["delta"].mean(), "min:", pop["delta"].min(), "max:", pop["delta"].max())
    print("kappa -> media:", pop["kappa"].mean(), "(debe ser 0 para todos)")
    print("correlación muestral delta-kappa:", np.corrcoef(pop["delta"], pop["kappa"])[0, 1])
