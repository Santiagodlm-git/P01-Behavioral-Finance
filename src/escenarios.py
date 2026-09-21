"""
Paso 6 -- Corredor de los 8 escenarios.

Este modulo no simula nada por si mismo: toma el motor de simulacion.py y lo
corre una vez por cada fila de la tabla de escenarios del enunciado, con la
poblacion regenerada en cada caso segun los valores inyectados de delta y kappa.

Produce dos tablas por escenario, que son el insumo de los Pasos 7 y 8:

  conteos_pgr_plr : una fila por cuenta con G_r, G_p, L_r, L_p  -> Paso 7
  resumen_cuentas : una fila por cuenta con rendimiento bruto, rendimiento
                    neto, turnover y los controles X_i                -> Paso 8

El universo de precios se genera UNA sola vez y se reutiliza en los 8
escenarios, de modo que cualquier diferencia entre escenarios sea atribuible
solo a los parametros de comportamiento inyectados, no a variacion de mercado.
"""

import os

import numpy as np
import pandas as pd

from generador_poblacion import generate_population
from generador_precios import generador_de_precios
from simulacion import simular_escenario, TASA_COSTO_TOTAL

# ---------------------------------------------------------------------------
# Parametros globales de la corrida
# ---------------------------------------------------------------------------
N_TRADERS = 1000
N_DAYS = 500
N_ASSETS = 50
PRECIO_INICIAL = 100.0
MU_ANNUAL = 0.08
SIGMA_ANNUAL = 0.20
SIGMA_MARKET_ANNUAL = 0.15

SEED_PRECIOS = 2024      # universo de precios: fijo para los 8 escenarios
SEED_POBLACION = 100     # se le suma el numero de escenario
SEED_DECISIONES = 500    # se le suma el numero de escenario
DIAS_POR_ANIO = 252

CARPETA_RESULTADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


# ---------------------------------------------------------------------------
# La tabla de escenarios del enunciado
# ---------------------------------------------------------------------------
ESCENARIOS = [
    {"num": 1, "nombre": "null",                 "delta": 0.0, "kappa": 0.0, "confound": None,
     "esperado": "Ambos estimadores callados"},
    {"num": 2, "nombre": "Disposition only low", "delta": 0.3, "kappa": 0.0, "confound": None,
     "esperado": "Disposition positivo, overconfidence callado"},
    {"num": 3, "nombre": "Disposition only high","delta": 0.8, "kappa": 0.0, "confound": None,
     "esperado": "Disposition positivo mayor, overconfidence callado"},
    {"num": 4, "nombre": "Turnover only low",    "delta": 0.0, "kappa": 0.3, "confound": None,
     "esperado": "Disposition callado, overconfidence negativo"},
    {"num": 5, "nombre": "Turnover only high",   "delta": 0.0, "kappa": 0.8, "confound": None,
     "esperado": "Disposition callado, overconfidence mas negativo"},
    {"num": 6, "nombre": "Both active",          "delta": 0.8, "kappa": 0.8, "confound": None,
     "esperado": "Revisar la interaccion entre ambos estimadores"},
    {"num": 7, "nombre": "Rebalancing confound", "delta": 0.0, "kappa": 0.0, "confound": "rebalanceo",
     "esperado": "Disposition probablemente se dispara de forma espuria"},
    {"num": 8, "nombre": "Mean-reversion confound", "delta": 0.0, "kappa": 0.0, "confound": "reversion",
     "esperado": "Disposition probablemente se dispara de forma espuria"},
]


def configuracion(num):
    """Devuelve la configuracion del escenario con ese numero."""
    for cfg in ESCENARIOS:
        if cfg["num"] == num:
            return cfg
    raise ValueError("No existe el escenario %r" % num)


# ---------------------------------------------------------------------------
# Utilidades de agregacion
# ---------------------------------------------------------------------------
def retorno_diario_de_mercado(df_prices):
    """
    Retorno diario del 'mercado', definido como el promedio equiponderado de
    los retornos simples de los 50 activos. Se usa como factor de referencia
    para estimar la beta de cada cuenta (la medida de exposicion al riesgo que
    pide el Paso 8 como control X_i).
    """
    precios = df_prices.values
    retornos_activos = precios[1:] / precios[:-1] - 1.0
    return retornos_activos.mean(axis=1)


def betas_por_cuenta(valor_cartera_diario, retorno_mercado):
    """
    Beta realizada de cada cuenta: covarianza entre el retorno diario de su
    cartera y el retorno del mercado, dividida entre la varianza del mercado.

    Se calcula sobre los retornos observados y no sobre las betas de los
    activos, para que la medida refleje la cartera que el agente sostuvo
    realmente a lo largo del tiempo, incluidas sus rotaciones.
    """
    V = valor_cartera_diario
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(V[:-1] > 0, V[1:] / np.where(V[:-1] > 0, V[:-1], 1.0) - 1.0, 0.0)
    m = retorno_mercado - retorno_mercado.mean()
    R_centrado = R - R.mean(axis=0, keepdims=True)
    var_m = float(m @ m)
    return (R_centrado.T @ m) / var_m


def resumen_por_cuenta(poblacion, resultado, df_prices):
    """
    Construye la tabla de una fila por cuenta que consume la regresion del
    Paso 8.

    Sobre bruto vs neto: el motor descuenta comision y spread del efectivo, de
    modo que el valor de cartera que reporta ya es NETO de costos. El
    rendimiento bruto se reconstruye sumando de vuelta los costos pagados.
    Hay que ser explicito sobre lo que esto implica: la diferencia entre bruto
    y neto queda ligada por construccion al monto operado, asi que la relacion
    beta_neto - beta_bruto = -0.003 funciona como prueba de consistencia de la
    contabilidad, NO como evidencia independiente. La prueba que si informa es
    si beta_bruto resulta indistinguible de cero.
    """
    N = len(poblacion["delta"])
    V = resultado["valor_cartera_diario"]
    valor_inicial = V[0]
    valor_final = V[-1]
    valor_promedio = V.mean(axis=0)

    t = resultado["transacciones"]
    idx = pd.RangeIndex(N, name="trader")
    if len(t):
        valor_operado = t["acciones"].abs() * t["precio"]
        aux = pd.DataFrame({"trader": t["trader"], "tipo": t["tipo"],
                            "valor": valor_operado, "costo": t["costo_usd"]})
        por_tipo = aux.pivot_table(index="trader", columns="tipo", values="valor",
                                   aggfunc="sum").reindex(idx).fillna(0.0)
        comprado = por_tipo["compra"].values if "compra" in por_tipo else np.zeros(N)
        vendido = por_tipo["venta"].values if "venta" in por_tipo else np.zeros(N)
        costos = aux.groupby("trader")["costo"].sum().reindex(idx).fillna(0.0).values
    else:
        comprado = vendido = costos = np.zeros(N)

    # Turnover del periodo (definicion de Barber & Odean) y su version anual.
    turnover_periodo = np.divide(comprado + vendido, 2.0 * valor_promedio,
                                 out=np.zeros(N), where=valor_promedio > 0)
    turnover_anual = turnover_periodo * (DIAS_POR_ANIO / N_DAYS)

    retorno_neto = valor_final / valor_inicial - 1.0
    retorno_bruto = (valor_final + costos) / valor_inicial - 1.0

    return pd.DataFrame({
        "trader": np.arange(N),
        "delta": poblacion["delta"],
        "kappa": poblacion["kappa"],
        "W": poblacion["W"],
        "n_objetivo": poblacion["n"],
        "retorno_bruto": retorno_bruto,
        "retorno_neto": retorno_neto,
        "turnover_periodo": turnover_periodo,
        "turnover_anual": turnover_anual,
        "valor_promedio_cartera": valor_promedio,
        "posiciones_promedio": resultado["posiciones_promedio"],
        "exposicion_riesgo": betas_por_cuenta(V, retorno_diario_de_mercado(df_prices)),
        "frac_efectivo_promedio": resultado["frac_efectivo_promedio"],
        "costos_usd": costos,
        "valor_comprado": comprado,
        "valor_vendido": vendido,
    })


def pgr_plr_agregado(conteos):
    """
    PGR y PLR agregados sobre toda la poblacion: se suman los conteos de todas
    las cuentas y hasta el final se divide, como en Odean (1998). Es un
    resumen rapido para inspeccionar la corrida; el estimador formal, con
    bootstrap por cuenta, vive en el Paso 7.
    """
    Gr, Gp = conteos["G_r"].sum(), conteos["G_p"].sum()
    Lr, Lp = conteos["L_r"].sum(), conteos["L_p"].sum()
    pgr = Gr / (Gr + Gp) if (Gr + Gp) else np.nan
    plr = Lr / (Lr + Lp) if (Lr + Lp) else np.nan
    return {"G_r": Gr, "G_p": Gp, "L_r": Lr, "L_p": Lp,
            "PGR": pgr, "PLR": plr, "PGR_menos_PLR": pgr - plr,
            "PGR_sobre_PLR": pgr / plr if plr else np.nan}


# ---------------------------------------------------------------------------
# Corrida de un escenario y de todos
# ---------------------------------------------------------------------------
def correr_escenario(num, df_prices, verbose=True):
    """
    Regenera la poblacion completa con la configuracion del escenario y corre
    el motor sobre el universo de precios compartido.
    """
    cfg = configuracion(num)

    if cfg["confound"] is not None:
        raise NotImplementedError(
            "El escenario %d requiere el confound '%s', que todavia no esta "
            "implementado en el motor." % (cfg["num"], cfg["confound"]))

    poblacion = generate_population(
        N=N_TRADERS, delta_center=cfg["delta"], kappa_center=cfg["kappa"],
        seed=SEED_POBLACION + cfg["num"])

    resultado = simular_escenario(
        poblacion, df_prices, seed_decisiones=SEED_DECISIONES + cfg["num"])

    conteos = resultado["conteos_pgr_plr"]
    resumen = resumen_por_cuenta(poblacion, resultado, df_prices)
    agregado = pgr_plr_agregado(conteos)

    # Correlaciones que pide el apartado de Independencia (Paso 10). La de
    # delta con kappa es indefinida cuando alguno es constante en cero.
    with np.errstate(invalid="ignore"):
        corr_dk = np.corrcoef(resumen["delta"], resumen["kappa"])[0, 1]
        corr_dt = np.corrcoef(resumen["delta"], resumen["turnover_anual"])[0, 1]
        corr_kt = np.corrcoef(resumen["kappa"], resumen["turnover_anual"])[0, 1]

    fila = {
        "escenario": cfg["num"], "nombre": cfg["nombre"],
        "delta_inyectado": cfg["delta"], "kappa_inyectado": cfg["kappa"],
        "confound": cfg["confound"] or "ninguno",
        **agregado,
        "turnover_anual_medio": resumen["turnover_anual"].mean(),
        "retorno_bruto_medio": resumen["retorno_bruto"].mean(),
        "retorno_neto_medio": resumen["retorno_neto"].mean(),
        "corr_delta_kappa": corr_dk,
        "corr_delta_turnover": corr_dt,
        "corr_kappa_turnover": corr_kt,
        "transacciones": len(resultado["transacciones"]),
    }

    if verbose:
        print("  esc %d (%-26s) PGR=%.4f PLR=%.4f dif=%+.4f  turnover=%.2f  "
              "bruto=%+.1f%% neto=%+.1f%%"
              % (cfg["num"], cfg["nombre"], agregado["PGR"], agregado["PLR"],
                 agregado["PGR_menos_PLR"], fila["turnover_anual_medio"],
                 100 * fila["retorno_bruto_medio"], 100 * fila["retorno_neto_medio"]))

    return {"config": cfg, "conteos": conteos, "resumen": resumen, "fila": fila}


def correr_todos(numeros=None, guardar=True, verbose=True):
    """
    Corre los escenarios indicados (por defecto los 8) y guarda las tablas en
    la carpeta results/. Los escenarios cuyo confound no este implementado se
    reportan como pendientes en vez de interrumpir la corrida.
    """
    numeros = numeros or [c["num"] for c in ESCENARIOS]

    df_prices = generador_de_precios(
        n_days=N_DAYS, n_assets=N_ASSETS, initial_price=PRECIO_INICIAL,
        mu_annual=MU_ANNUAL, sigma_annual=SIGMA_ANNUAL,
        sigma_market_annual=SIGMA_MARKET_ANNUAL, seed=SEED_PRECIOS)

    if guardar:
        os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

    filas, pendientes = [], []
    for num in numeros:
        try:
            salida = correr_escenario(num, df_prices, verbose=verbose)
        except NotImplementedError as e:
            pendientes.append((num, str(e)))
            if verbose:
                print("  esc %d PENDIENTE: %s" % (num, e))
            continue

        filas.append(salida["fila"])
        if guardar:
            salida["conteos"].to_csv(
                os.path.join(CARPETA_RESULTADOS, "conteos_pgr_plr_esc%d.csv" % num), index=False)
            salida["resumen"].to_csv(
                os.path.join(CARPETA_RESULTADOS, "resumen_cuentas_esc%d.csv" % num), index=False)

    tabla = pd.DataFrame(filas)
    if guardar and len(tabla):
        tabla.to_csv(os.path.join(CARPETA_RESULTADOS, "tabla_escenarios.csv"), index=False)

    return tabla, pendientes


if __name__ == "__main__":
    print("=== Paso 6: corrida de los escenarios ===")
    print("Poblacion: %d traders | Mercado: %d activos x %d dias | costo total %.2f bps\n"
          % (N_TRADERS, N_ASSETS, N_DAYS, TASA_COSTO_TOTAL * 10_000))

    tabla, pendientes = correr_todos()

    print("\n--- Tabla resumen ---")
    cols = ["escenario", "delta_inyectado", "kappa_inyectado", "PGR", "PLR",
            "PGR_menos_PLR", "turnover_anual_medio", "corr_delta_turnover"]
    print(tabla[cols].to_string(index=False, float_format=lambda x: "%.4f" % x))

    if pendientes:
        print("\n--- Escenarios pendientes ---")
        for num, motivo in pendientes:
            print("  %d: %s" % (num, motivo))
