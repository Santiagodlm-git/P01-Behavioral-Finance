import numpy as np


def utilidad_valor(x, alpha=0.88, beta_exp=0.88, lam=2.25):
    """
    Funcion de valor de Kahneman-Tversky (forma de S).
    x: ganancia (x>=0) o perdida (x<0) respecto al precio de compra.
    alpha, beta_exp: exponentes que dan la concavidad/convexidad (tipicamente ~0.88).
    lam: aversion a la perdida (tipicamente ~2.25), aplica solo al lado de perdidas.

    Nota: alpha, beta_exp y lam son constantes de la psicologia humana
    (iguales para todos los traders) -- NO varian por trader.
    Lo que varia por trader es delta_i, que se aplica despues, en decide_venta.
    """
    x = np.asarray(x, dtype=float)
    x_pos = np.clip(x, 0, None)   # evita elevar numeros negativos a potencia fraccionaria
    x_neg = np.clip(-x, 0, None)
    ganancia = np.where(x >= 0, x_pos ** alpha, 0.0)
    perdida = np.where(x < 0, -lam * (x_neg ** beta_exp), 0.0)
    return ganancia + perdida


def decide_venta(precio_actual, precio_compra, delta_i, rng,
                  mu_annual, sigma_annual, dt=1/252,
                  alpha=0.88, beta_exp=0.88, lam=2.25, temperatura=0.10, h0=0.02):
    """
    Decide si un agente vende una posicion, usando utilidad de punto de referencia.

    precio_actual, precio_compra: escalares o arrays del mismo tamaño.
    delta_i: fuerza de disposition effect del agente (0 a 1).
    rng: generador de numeros aleatorios (para el paso probabilistico final).
    temperatura: controla que tan abrupta es la transicion sí/no alrededor
        del punto de indiferencia. Mas baja = mas parecido a un interruptor duro.

    Regresa: array booleano, True = vende.
    """
    precio_actual = np.asarray(precio_actual, dtype=float)
    precio_compra = np.asarray(precio_compra, dtype=float)

    # --- Utilidad de vender AHORA (sobre el RENDIMIENTO, no sobre dolares) ---
    r_ahora = precio_actual / precio_compra - 1
    v_ahora = utilidad_valor(r_ahora, alpha, beta_exp, lam)
    v_vender = (1 + delta_i) * v_ahora

    # --- Valor de CONTINUAR (un dia hacia adelante, CON incertidumbre) ---
    diffusion_diaria = sigma_annual * np.sqrt(dt)
    precio_sube = precio_actual * np.exp(mu_annual * dt + diffusion_diaria)
    precio_baja = precio_actual * np.exp(mu_annual * dt - diffusion_diaria)
    v_sube = utilidad_valor(precio_sube / precio_compra - 1, alpha, beta_exp, lam)
    v_baja = utilidad_valor(precio_baja / precio_compra - 1, alpha, beta_exp, lam)
    v_continuar = 0.5 * v_sube + 0.5 * v_baja

    # --- Decision PROBABILISTICA (logit), en vez de un interruptor duro ---
    # Sin esto, cualquier delta>0 no trivial satura la decision a 100%/0% y se
    # pierde la relacion graduada entre delta y el comportamiento (falla de
    # monotonicidad). La funcion logistica traduce la BRECHA de utilidad en
    # una probabilidad suave; temperatura controla que tan "dura" es esa curva.
    brecha = v_vender - v_continuar
    prob_vende = 2 * h0 / (1 + np.exp(-brecha / temperatura))

    return rng.random(size=prob_vende.shape) < prob_vende


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Ejemplo minimo: un trader con delta=0.8, dos posiciones (ganadora, perdedora)
    precio_actual = np.array([120.0, 80.0])
    precio_compra = np.array([100.0, 100.0])
    resultado = decide_venta(precio_actual, precio_compra, 0.8, rng,
                              mu_annual=0.08, sigma_annual=0.20)
    print("Caso minimo -> ¿Vende?", resultado, "(esperado: sesgo hacia [True, False])")
    print()

    # Prueba de graduacion: ¿la brecha PGR-PLR crece con delta?
    rng2 = np.random.default_rng(7)
    n = 20000
    pct_change = rng2.uniform(0.01, 0.30, size=n)
    precio_compra2 = np.full(n, 100.0)
    precio_ganador2 = precio_compra2 * (1 + pct_change)
    precio_perdedor2 = precio_compra2 * (1 - pct_change)

    print("--- Graduacion de la brecha PGR-like / PLR-like segun delta ---")
    for delta in [0.0, 0.1, 0.3, 0.5, 0.8]:
        rng_g = np.random.default_rng(100)
        rng_p = np.random.default_rng(200)
        vg = decide_venta(precio_ganador2, precio_compra2, delta, rng_g, mu_annual=0.08, sigma_annual=0.20)
        vp = decide_venta(precio_perdedor2, precio_compra2, delta, rng_p, mu_annual=0.08, sigma_annual=0.20)
        print(f"delta={delta}: % vende ganadoras={vg.mean()*100:.1f}%  "
              f"% vende perdedoras={vp.mean()*100:.1f}%  brecha={(vg.mean()-vp.mean())*100:.1f} pts")
