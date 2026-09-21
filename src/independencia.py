"""
Paso 10 -- Independencia de delta y kappa.

El enunciado pide cuatro cosas, y este modulo produce las cuatro:

  1. Sortear delta_i y kappa_i de forma independiente.
  2. Reportar la correlacion muestral realizada entre ambos parametros.
  3. Reportar la correlacion entre delta_i y el turnover MEDIDO, que no sera
     cero aunque (1) se cumpla, y explicar por que.
  4. Discutir que pasaria con el estimador de sobreconfianza si los parametros
     se hubieran generado correlacionados. Aqui no se discute en abstracto: se
     genera una poblacion con correlacion inducida y se estima, para mostrar
     cuanto se mueve el coeficiente.

El diseno experimental separa el efecto de cada parametro fijando el otro para
toda la poblacion. Mirar las correlaciones dentro de un escenario mezcla los dos
canales; fijar uno los aisla.
"""

import os

import numpy as np
import pandas as pd

from generador_poblacion import generate_population
from generador_precios import generador_de_precios
from simulacion import simular_escenario
from escenarios import (resumen_por_cuenta, N_TRADERS, N_DAYS, N_ASSETS,
                         PRECIO_INICIAL, MU_ANNUAL, SIGMA_ANNUAL,
                         SIGMA_MARKET_ANNUAL, SEED_PRECIOS, CARPETA_RESULTADOS)
from regresion import ols_hc1, matriz_diseno, CONTROLES

SEED_BASE = 3000


# ---------------------------------------------------------------------------
# Construccion de poblaciones controladas
# ---------------------------------------------------------------------------
def poblacion_controlada(delta_center, kappa_center, fijar, seed):
    """
    Genera una poblacion donde uno de los dos parametros se mantiene CONSTANTE
    para todos los agentes y el otro varia.

    fijar: "delta" o "kappa" -- cual queda constante. El otro conserva la
    dispersion que produce generate_population, de modo que la unica fuente de
    variacion entre cuentas es ese parametro (ademas de W y n, que se controlan
    en la regresion).
    """
    pob = generate_population(N=N_TRADERS, delta_center=delta_center,
                               kappa_center=kappa_center, seed=seed)
    pob = {k: v.copy() for k, v in pob.items()}
    if fijar == "delta":
        pob["delta"] = np.full(N_TRADERS, delta_center)
    elif fijar == "kappa":
        pob["kappa"] = np.full(N_TRADERS, kappa_center)
    elif fijar is not None:
        raise ValueError("fijar debe ser 'delta', 'kappa' o None")
    return pob


def poblacion_correlacionada(delta_center, kappa_center, rho_objetivo, seed):
    """
    Genera una poblacion donde kappa se construye a partir de delta mas ruido,
    de modo que ambos parametros quedan correlacionados por construccion.

    Sirve para el punto 4 del enunciado: mostrar empiricamente que pasa con el
    estimador de sobreconfianza cuando se viola el supuesto de independencia.
    """
    pob = poblacion_controlada(delta_center, kappa_center, fijar=None, seed=seed)
    rng = np.random.default_rng(seed + 99)

    d = pob["delta"]
    z = (d - d.mean()) / d.std()
    ruido = rng.standard_normal(N_TRADERS)
    mezcla = rho_objetivo * z + np.sqrt(max(1 - rho_objetivo ** 2, 0.0)) * ruido

    # Se lleva la mezcla al rango y la dispersion de la kappa original,
    # conservando su media, para que el escenario siga siendo comparable.
    k = pob["kappa"]
    pob["kappa"] = np.clip(k.mean() + mezcla * k.std(), 0.0, 1.0)
    return pob


# ---------------------------------------------------------------------------
# Medicion
# ---------------------------------------------------------------------------
def medir(pob, df_prices, seed_decisiones):
    """Corre una poblacion y devuelve el resumen por cuenta con extras."""
    res = simular_escenario(pob, df_prices, seed_decisiones=seed_decisiones)
    resumen = resumen_por_cuenta(pob, res, df_prices)
    resumen["log_cartera"] = np.log(resumen["valor_promedio_cartera"].clip(lower=1.0))

    t = res["transacciones"]
    ventas = t[t["tipo"] == "venta"].groupby("trader").size()
    resumen["n_ventas"] = ventas.reindex(range(len(resumen)), fill_value=0).to_numpy()

    # Fraccion de las posiciones vivas al final que estan por debajo de su
    # precio de compra: es el mecanismo por el que delta frena la rotacion.
    ocupado = res["held_asset"] >= 0
    precio_final = df_prices.values[-1][np.where(ocupado, res["held_asset"], 0)]
    perdedora = ocupado & (precio_final < res["held_price"])
    with np.errstate(invalid="ignore"):
        resumen["pct_perdedoras_final"] = np.divide(
            perdedora.sum(axis=1), ocupado.sum(axis=1),
            out=np.zeros(len(resumen)), where=ocupado.sum(axis=1) > 0)
    return resumen


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def beta_turnover(resumen, columna="retorno_neto"):
    """Coeficiente del turnover en la regresion del Paso 8."""
    X, nombres = matriz_diseno(resumen, CONTROLES)
    r = ols_hc1(resumen[columna].to_numpy(dtype=float), X, nombres)
    return r["beta"][1], r["se"][1], r["t"][1]


# ---------------------------------------------------------------------------
# Las cuatro pruebas del enunciado
# ---------------------------------------------------------------------------
def correr_pruebas(verbose=True):
    df_prices = generador_de_precios(
        n_days=N_DAYS, n_assets=N_ASSETS, initial_price=PRECIO_INICIAL,
        mu_annual=MU_ANNUAL, sigma_annual=SIGMA_ANNUAL,
        sigma_market_annual=SIGMA_MARKET_ANNUAL, seed=SEED_PRECIOS)

    filas = []

    # (1) y (2): ambos parametros varian -> correlacion realizada entre ellos
    pob_ambos = poblacion_controlada(0.5, 0.5, fijar=None, seed=SEED_BASE + 1)
    r_ambos = medir(pob_ambos, df_prices, SEED_BASE + 1)
    filas.append({
        "prueba": "Ambos varian",
        "corr_delta_kappa": corr(r_ambos["delta"], r_ambos["kappa"]),
        "corr_delta_turnover": corr(r_ambos["delta"], r_ambos["turnover_anual"]),
        "corr_kappa_turnover": corr(r_ambos["kappa"], r_ambos["turnover_anual"]),
    })

    # (3a): efecto puro de delta -> kappa constante
    pob_d = poblacion_controlada(0.5, 0.3, fijar="kappa", seed=SEED_BASE + 2)
    r_d = medir(pob_d, df_prices, SEED_BASE + 2)
    filas.append({
        "prueba": "Solo varia delta (kappa fijo)",
        "corr_delta_kappa": np.nan,
        "corr_delta_turnover": corr(r_d["delta"], r_d["turnover_anual"]),
        "corr_kappa_turnover": np.nan,
    })

    # (3b): efecto puro de kappa -> delta constante
    pob_k = poblacion_controlada(0.4, 0.5, fijar="delta", seed=SEED_BASE + 3)
    r_k = medir(pob_k, df_prices, SEED_BASE + 3)
    filas.append({
        "prueba": "Solo varia kappa (delta fijo)",
        "corr_delta_kappa": np.nan,
        "corr_delta_turnover": np.nan,
        "corr_kappa_turnover": corr(r_k["kappa"], r_k["turnover_anual"]),
    })

    tabla = pd.DataFrame(filas)

    # Mecanismo: el turnover es un cociente, asi que delta puede moverlo por el
    # numerador (cuanto valor se rota) o por el denominador (que tan grande es
    # la cartera). Se miden los dos, y el conteo de operaciones por separado,
    # para no atribuirle el efecto al canal equivocado.
    q = r_d["delta"].rank(pct=True)
    bajo, alto = r_d[q < 0.25], r_d[q > 0.75]
    mecanismo = pd.DataFrame({
        "medida": ["delta", "turnover anual", "numero de ventas", "valor vendido (USD)",
                    "valor promedio de cartera (USD)", "% posiciones perdedoras al final"],
        "columna": ["delta", "turnover_anual", "n_ventas", "valor_vendido",
                     "valor_promedio_cartera", "pct_perdedoras_final"],
    })
    mecanismo["cuartil_delta_bajo"] = [bajo[c].mean() for c in mecanismo["columna"]]
    mecanismo["cuartil_delta_alto"] = [alto[c].mean() for c in mecanismo["columna"]]
    mecanismo["cambio_pct"] = 100 * (mecanismo["cuartil_delta_alto"] /
                                      mecanismo["cuartil_delta_bajo"] - 1)
    mecanismo["corr_con_delta"] = [corr(r_d["delta"], r_d[c]) for c in mecanismo["columna"]]
    mecanismo = mecanismo.drop(columns=["columna"])

    # (4): que pasa si los parametros se generan correlacionados
    def fila_comparacion(etiqueta, resumen):
        bb, seb, tb = beta_turnover(resumen, "retorno_bruto")
        bn, sen, tn = beta_turnover(resumen, "retorno_neto")
        return {"poblacion": etiqueta,
                "corr_delta_kappa": corr(resumen["delta"], resumen["kappa"]),
                "beta_bruto": bb, "t_bruto": tb,
                "beta_neto": bn, "t_neto": tn}

    comparacion = [fila_comparacion("delta y kappa independientes", r_ambos)]
    for rho in (0.5, 0.8):
        pob_c = poblacion_correlacionada(0.5, 0.5, rho, seed=SEED_BASE + 10)
        r_c = medir(pob_c, df_prices, SEED_BASE + 10)
        comparacion.append(fila_comparacion("correlacion inducida rho = %.1f" % rho, r_c))
    comparacion = pd.DataFrame(comparacion)

    if verbose:
        print("=" * 76)
        print("PASO 10 -- INDEPENDENCIA DE DELTA Y KAPPA")
        print("=" * 76)
        print("\n(1) y (2) Correlaciones realizadas\n")
        print(tabla.to_string(index=False, float_format=lambda x: "%+.4f" % x))
        print("\n(3) Mecanismo: por donde mueve delta al turnover\n")
        print(mecanismo.to_string(index=False, float_format=lambda x: "%+.4f" % x))
        print("\n(4) Que pasa si los parametros se generan correlacionados\n")
        print(comparacion.to_string(index=False, float_format=lambda x: "%+.5f" % x))

    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    tabla.to_csv(os.path.join(CARPETA_RESULTADOS, "independencia_correlaciones.csv"), index=False)
    mecanismo.to_csv(os.path.join(CARPETA_RESULTADOS, "independencia_mecanismo.csv"), index=False)
    comparacion.to_csv(os.path.join(CARPETA_RESULTADOS, "independencia_correlacion_inducida.csv"), index=False)
    return tabla, mecanismo, comparacion


if __name__ == "__main__":
    correr_pruebas()
