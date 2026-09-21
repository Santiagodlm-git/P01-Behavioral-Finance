"""
Paso 9 -- Diagnostico de las cuatro causas no conductuales.

El enunciado enumera cuatro razones por las que la pendiente bruta de la
regresion de turnover puede ser distinta de cero sin que haya informacion:
fuga de informacion, cash drag, spread dentro de la trayectoria de precios y
causalidad reversa. Este modulo reune la evidencia de cada una sobre el motor
actual y deja el veredicto por escrito.

La primera es un error y hay que corregirla; las otras tres son legitimas y hay
que documentarlas.
"""

import os

import numpy as np
import pandas as pd

from generador_precios import generador_de_precios
from generador_poblacion import generate_population
from simulacion import simular_escenario
from escenarios import (resumen_por_cuenta, configuracion, N_TRADERS, N_DAYS,
                         N_ASSETS, PRECIO_INICIAL, MU_ANNUAL, SIGMA_ANNUAL,
                         SIGMA_MARKET_ANNUAL, SEED_PRECIOS, SEED_POBLACION,
                         SEED_DECISIONES, CARPETA_RESULTADOS)
from regresion import cargar_escenario, regresion_rendimiento, CONTROLES


def evidencia_fuga_de_informacion():
    """
    Prueba 1 -- Si hubiera informacion filtrada, el rendimiento BRUTO promedio
    subiria con la actividad. Se compara el bruto medio de los 8 escenarios,
    cuyo turnover va de 2.0 a 6.5 veces al año.
    """
    tabla = pd.read_csv(os.path.join(CARPETA_RESULTADOS, "tabla_escenarios.csv"))
    rango = tabla["retorno_bruto_medio"]
    return {
        "prueba": "Fuga de informacion",
        "evidencia": ("bruto medio entre %.1f%% y %.1f%% mientras el turnover va de "
                       "%.1f a %.1f veces al año"
                       % (100 * rango.min(), 100 * rango.max(),
                          tabla["turnover_anual_medio"].min(),
                          tabla["turnover_anual_medio"].max())),
        "veredicto": "Descartada por diseno",
    }


def evidencia_cash_drag(df_prices, num=5):
    """
    Prueba 2 -- Se mide el efectivo ocioso real, dia por dia, y su relacion con
    el turnover. Si hubiera cash drag, las cuentas que mas operan pasarian mas
    tiempo fuera del mercado.
    """
    cfg = configuracion(num)
    pob = generate_population(N=N_TRADERS, delta_center=cfg["delta"],
                               kappa_center=cfg["kappa"], seed=SEED_POBLACION + num)
    res = simular_escenario(pob, df_prices, seed_decisiones=SEED_DECISIONES + num,
                             confound=cfg["confound"])
    cash, valor = res["cash_diario"], res["valor_cartera_diario"]
    pct_agregado = cash.sum(axis=1) / valor.sum(axis=1)
    resumen = resumen_por_cuenta(pob, res, df_prices)
    c = np.corrcoef(resumen["turnover_periodo"], resumen["frac_efectivo_promedio"])[0, 1]
    serie = pd.DataFrame({"dia": [10, 100, 250, 400, N_DAYS],
                           "pct_efectivo": [100 * pct_agregado[d] for d in
                                            (10, 100, 250, 400, N_DAYS)]})
    return {
        "prueba": "Cash drag",
        "evidencia": ("efectivo agregado entre %.2f%% y %.2f%% del dia 10 en adelante; "
                       "corr(turnover, efectivo) = %+.3f"
                       % (pct_agregado[10:].min() * 100, pct_agregado[10:].max() * 100, c)),
        "veredicto": "Descartada por diseno",
    }, serie


def evidencia_spread():
    """
    Prueba 3 -- El spread no vive dentro de la trayectoria de precios: toda
    ejecucion ocurre al precio medio y el costo se registra aparte. La prueba
    es que la diferencia entre el rendimiento bruto y el neto reproduce
    exactamente la tarifa.
    """
    tabla = pd.read_csv(os.path.join(CARPETA_RESULTADOS, "tabla_overconfidence.csv"))
    return {
        "prueba": "Spread dentro del precio",
        "evidencia": ("beta neto menos beta bruto entre %+.5f y %+.5f contra el "
                       "-0.00300 que predicen los 15 bps por lado"
                       % (tabla["diferencia"].min(), tabla["diferencia"].max())),
        "veredicto": "Descartada por diseno",
    }


def evidencia_causalidad_reversa():
    """
    Prueba 4 -- La causalidad reversa deberia aparecer solo donde la regla de
    venta condiciona sobre el precio de compra o sobre el movimiento del
    precio, y no donde la variacion de turnover viene de kappa.
    """
    tabla = pd.read_csv(os.path.join(CARPETA_RESULTADOS, "tabla_overconfidence.csv"))
    filas = []
    for _, r in tabla.iterrows():
        num = int(r["escenario"])
        cfg = configuracion(num)
        condiciona = (cfg["delta"] > 0) or (cfg["confound"] == "reversion")
        filas.append({
            "escenario": num,
            "regla_condiciona_en_precio_de_compra": "si" if condiciona else "no",
            "beta_bruto": r["beta_bruto"], "t_bruto": r["t_bruto"],
            "significativo": "si" if abs(r["t_bruto"]) > 2 else "no",
        })
    detalle = pd.DataFrame(filas)
    con = detalle[detalle["regla_condiciona_en_precio_de_compra"] == "si"]
    sin = detalle[detalle["regla_condiciona_en_precio_de_compra"] == "no"]
    return {
        "prueba": "Causalidad reversa",
        "evidencia": ("beta bruto significativo en %d de %d escenarios cuya regla "
                       "condiciona en el precio, y en %d de %d donde no"
                       % ((con["significativo"] == "si").sum(), len(con),
                          (sin["significativo"] == "si").sum(), len(sin))),
        "veredicto": "PRESENTE, y es propiedad del estimador, no del simulador",
    }, detalle


def correr(verbose=True):
    df_prices = generador_de_precios(
        n_days=N_DAYS, n_assets=N_ASSETS, initial_price=PRECIO_INICIAL,
        mu_annual=MU_ANNUAL, sigma_annual=SIGMA_ANNUAL,
        sigma_market_annual=SIGMA_MARKET_ANNUAL, seed=SEED_PRECIOS)

    f1 = evidencia_fuga_de_informacion()
    f2, serie_cash = evidencia_cash_drag(df_prices)
    f3 = evidencia_spread()
    f4, detalle_reversa = evidencia_causalidad_reversa()
    tabla = pd.DataFrame([f1, f2, f3, f4])

    if verbose:
        print("=" * 76)
        print("PASO 9 -- LAS CUATRO CAUSAS NO CONDUCTUALES")
        print("=" * 76)
        for _, r in tabla.iterrows():
            print("\n%s" % r["prueba"])
            print("   evidencia: %s" % r["evidencia"])
            print("   veredicto: %s" % r["veredicto"])
        print("\nEfectivo agregado a lo largo del escenario 5 (el mas activo):")
        print(serie_cash.to_string(index=False, float_format=lambda x: "%.2f" % x))
        print("\nDonde aparece la causalidad reversa:")
        print(detalle_reversa.to_string(index=False, float_format=lambda x: "%.5f" % x))

    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    tabla.to_csv(os.path.join(CARPETA_RESULTADOS, "diagnostico_trampas.csv"), index=False)
    detalle_reversa.to_csv(os.path.join(CARPETA_RESULTADOS, "diagnostico_reversa.csv"), index=False)
    return tabla, serie_cash, detalle_reversa


if __name__ == "__main__":
    correr()
