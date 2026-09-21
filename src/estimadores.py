"""
Paso 7 -- Estimador del disposition effect (PGR / PLR).

Toma los conteos por cuenta que produce el motor y calcula el estimador de
Odean (1998), con errores estandar por bootstrap POR CUENTA.

Las cuatro cajas, contadas solo en los dias en que la cuenta vendio algo:

    G_r : ganancias realizadas      G_p : ganancias en papel (retenidas)
    L_r : perdidas realizadas       L_p : perdidas en papel  (retenidas)

            PGR = G_r / (G_r + G_p)        PLR = L_r / (L_r + L_p)

Declaraciones exigidas por el enunciado, resueltas aqui de forma explicita:

  * Dias sin ventas: no aportan a ningun conteo. La clasificacion ocurre en el
    motor y solo se dispara cuando la cuenta realiza al menos una operacion.
  * Ventas parciales: cuentan como realizacion en la medida principal, y se
    llevan aparte para poder recalcular ambos estimadores excluyendolas.
  * Costo base: cada posicion se abre con una sola compra, asi que FIFO, costo
    promedio y por lote coinciden. La unica excepcion son las reposiciones del
    rebalanceo, donde el motor usa costo promedio ponderado.
  * Posicion exactamente en su precio de compra: se cuenta aparte (columna
    'empates') y se excluye de los cuatro conteos, porque no es ni ganancia ni
    perdida. Con precios continuos es un evento de medida practicamente nula.
"""

import os

import numpy as np
import pandas as pd

CARPETA_RESULTADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


# ---------------------------------------------------------------------------
# Bloque 1 -- El estimador puntual
# ---------------------------------------------------------------------------
def cajas(conteos, excluir_parciales=False):
    """
    Devuelve los cuatro conteos por cuenta como arreglos de numpy.

    Con excluir_parciales=True se restan las realizaciones que vinieron de un
    recorte parcial. Es la prueba de robustez de Odean: si el efecto observado
    desaparece al quitar las parciales, viene de rebalanceo y no de una
    preferencia por realizar ganancias.
    """
    Gr = conteos["G_r"].to_numpy(dtype=float)
    Gp = conteos["G_p"].to_numpy(dtype=float)
    Lr = conteos["L_r"].to_numpy(dtype=float)
    Lp = conteos["L_p"].to_numpy(dtype=float)

    if excluir_parciales:
        Gr = Gr - conteos["G_r_parcial"].to_numpy(dtype=float)
        Lr = Lr - conteos["L_r_parcial"].to_numpy(dtype=float)

    return Gr, Gp, Lr, Lp


def pgr_plr_desde_cajas(Gr, Gp, Lr, Lp):
    """
    Agrega los conteos de TODAS las cuentas y hasta el final divide.

    El orden importa: sumar primero y dividir despues no es lo mismo que
    promediar los PGR de cada cuenta. Odean usa el agregado, que pondera a cada
    cuenta por su numero de observaciones.
    """
    sGr, sGp, sLr, sLp = Gr.sum(), Gp.sum(), Lr.sum(), Lp.sum()
    pgr = sGr / (sGr + sGp) if (sGr + sGp) > 0 else np.nan
    plr = sLr / (sLr + sLp) if (sLr + sLp) > 0 else np.nan
    return pgr, plr


# ---------------------------------------------------------------------------
# Bloque 2 -- Bootstrap POR CUENTA
# ---------------------------------------------------------------------------
def bootstrap_por_cuenta(Gr, Gp, Lr, Lp, n_replicas=1000, seed=0):
    """
    Remuestrea CUENTAS con reemplazo y recalcula el estimador en cada replica.

    Por que por cuenta y no por transaccion: los conteos de una misma cuenta
    provienen del mismo agente, con el mismo delta y la misma cartera, asi que
    no son observaciones independientes. Tratarlas como si lo fueran subestima
    el error estandar por un factor grande. Odean discute exactamente este
    punto al senalar que los PGR por cuenta son heterocedasticos.

    Devuelve las distribuciones bootstrap de PGR, PLR, su diferencia y su razon.
    """
    rng = np.random.default_rng(seed)
    n = len(Gr)

    # Una matriz (n_replicas x n) de indices de cuentas sorteados con reemplazo.
    # Cada renglon es una poblacion sintetica del mismo tamano que la original.
    idx = rng.integers(0, n, size=(n_replicas, n))

    # Al indexar con esa matriz y sumar por renglon obtenemos, de un golpe, los
    # cuatro agregados de cada una de las 1,000 replicas.
    sGr = Gr[idx].sum(axis=1)
    sGp = Gp[idx].sum(axis=1)
    sLr = Lr[idx].sum(axis=1)
    sLp = Lp[idx].sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        pgr = np.where((sGr + sGp) > 0, sGr / (sGr + sGp), np.nan)
        plr = np.where((sLr + sLp) > 0, sLr / (sLr + sLp), np.nan)
        razon = np.where(plr > 0, pgr / plr, np.nan)

    return {"PGR": pgr, "PLR": plr, "diferencia": pgr - plr, "razon": razon}


def estimar_disposition(conteos, excluir_parciales=False, n_replicas=1000, seed=0):
    """
    Estimador completo: valor puntual, error estandar y intervalo de confianza
    al 95% por percentiles de la distribucion bootstrap.
    """
    Gr, Gp, Lr, Lp = cajas(conteos, excluir_parciales)
    pgr, plr = pgr_plr_desde_cajas(Gr, Gp, Lr, Lp)
    reps = bootstrap_por_cuenta(Gr, Gp, Lr, Lp, n_replicas, seed)

    salida = {
        "PGR": pgr, "PLR": plr,
        "diferencia": pgr - plr,
        "razon": pgr / plr if plr > 0 else np.nan,
        "G_r": Gr.sum(), "G_p": Gp.sum(), "L_r": Lr.sum(), "L_p": Lp.sum(),
        "cuentas": len(Gr),
        "cuentas_sin_observaciones": int(((Gr + Gp + Lr + Lp) == 0).sum()),
    }
    for nombre, muestras in reps.items():
        salida["se_" + nombre] = np.nanstd(muestras, ddof=1)
        salida["ic95_bajo_" + nombre] = np.nanpercentile(muestras, 2.5)
        salida["ic95_alto_" + nombre] = np.nanpercentile(muestras, 97.5)
    return salida


# ---------------------------------------------------------------------------
# Bloque 3 -- Validacion contra datos sinteticos de respuesta conocida
# ---------------------------------------------------------------------------
def conteos_sinteticos(n_cuentas=400, n_dias=250, n_ganancias=12, n_perdidas=8,
                        p_gana=0.06, p_pierde=0.03, seed=0):
    """
    Fabrica conteos con una respuesta que conocemos de antemano.

    Aqui SI se fija la probabilidad de venta a mano (p_gana y p_pierde). Eso no
    contradice la regla del enunciado de no parametrizar PGR directamente: no
    estamos simulando comportamiento, estamos probando una calculadora. El
    simulador de agentes sigue produciendo PGR y PLR como propiedades
    emergentes; este bloque solo verifica que la formula y el bootstrap esten
    bien programados.

    Cada cuenta tiene una cartera fija de n_ganancias posiciones arriba de su
    precio de compra y n_perdidas abajo. Cada dia se sortea cuales se venden, y
    los conteos solo se acumulan en los dias con al menos una venta.
    """
    rng = np.random.default_rng(seed)

    vend_g = rng.binomial(n_ganancias, p_gana, size=(n_dias, n_cuentas))
    vend_l = rng.binomial(n_perdidas, p_pierde, size=(n_dias, n_cuentas))
    hubo_venta = (vend_g + vend_l) > 0   # la condicion clave del estimador

    Gr = (vend_g * hubo_venta).sum(axis=0)
    Gp = ((n_ganancias - vend_g) * hubo_venta).sum(axis=0)
    Lr = (vend_l * hubo_venta).sum(axis=0)
    Lp = ((n_perdidas - vend_l) * hubo_venta).sum(axis=0)

    return pd.DataFrame({
        "trader": np.arange(n_cuentas),
        "G_r": Gr, "G_p": Gp, "L_r": Lr, "L_p": Lp,
        "empates": 0, "G_r_parcial": 0, "L_r_parcial": 0,
    })


def valor_teorico(n_ganancias, n_perdidas, p_gana, p_pierde):
    """
    Lo que PGR y PLR deberian dar, dado que solo se cuentan los dias con venta.

    P(ningun dia de venta) = (1-p_gana)^n_ganancias * (1-p_pierde)^n_perdidas

    Condicionar en 'hubo al menos una venta' infla ambas proporciones por el
    mismo factor 1/P(al menos una venta). Por eso los NIVELES de PGR y PLR
    dependen del tamano y la actividad de la cartera, mientras que la RAZON
    PGR/PLR se limpia de ese factor y recupera exactamente p_gana/p_pierde.
    """
    p_sin_venta = (1 - p_gana) ** n_ganancias * (1 - p_pierde) ** n_perdidas
    p_con_venta = 1 - p_sin_venta
    return p_gana / p_con_venta, p_pierde / p_con_venta, p_gana / p_pierde


# ---------------------------------------------------------------------------
# Bloque 4 -- Aplicacion a los 8 escenarios
# ---------------------------------------------------------------------------
def tabla_disposition(numeros=range(1, 9), n_replicas=1000, seed=0):
    """
    Corre el estimador sobre los conteos guardados por escenarios.py y arma la
    tabla de resultados con errores estandar.
    """
    filas = []
    for num in numeros:
        ruta = os.path.join(CARPETA_RESULTADOS, "conteos_pgr_plr_esc%d.csv" % num)
        if not os.path.exists(ruta):
            continue
        conteos = pd.read_csv(ruta)

        est = estimar_disposition(conteos, n_replicas=n_replicas, seed=seed)
        fila = {"escenario": num, "PGR": est["PGR"], "PLR": est["PLR"],
                "dif": est["diferencia"], "se_dif": est["se_diferencia"],
                "ic95_bajo": est["ic95_bajo_diferencia"],
                "ic95_alto": est["ic95_alto_diferencia"],
                "razon": est["razon"], "se_razon": est["se_razon"]}

        # Si el escenario trae realizaciones parciales, se agrega la version
        # que las excluye.
        if conteos["G_r_parcial"].sum() + conteos["L_r_parcial"].sum() > 0:
            sin = estimar_disposition(conteos, excluir_parciales=True,
                                      n_replicas=n_replicas, seed=seed)
            fila["dif_sin_parciales"] = sin["diferencia"]
            fila["se_dif_sin_parciales"] = sin["se_diferencia"]
        else:
            fila["dif_sin_parciales"] = fila["dif"]
            fila["se_dif_sin_parciales"] = fila["se_dif"]

        filas.append(fila)
    return pd.DataFrame(filas)


if __name__ == "__main__":
    print("=" * 74)
    print("VALIDACION DEL ESTIMADOR CONTRA DATOS SINTETICOS")
    print("=" * 74)
    print("Se fabrican conteos donde la respuesta se conoce de antemano.")
    print("Si el estimador no la recupera, el error esta en el estimador.\n")

    for p_g, p_l in [(0.06, 0.03), (0.04, 0.04), (0.10, 0.02)]:
        datos = conteos_sinteticos(p_gana=p_g, p_pierde=p_l, seed=1)
        est = estimar_disposition(datos, n_replicas=500, seed=1)
        t_pgr, t_plr, t_razon = valor_teorico(12, 8, p_g, p_l)
        print("  p_gana=%.2f  p_pierde=%.2f" % (p_g, p_l))
        print("     PGR   estimado %.4f   teorico %.4f" % (est["PGR"], t_pgr))
        print("     PLR   estimado %.4f   teorico %.4f" % (est["PLR"], t_plr))
        print("     razon estimada %.4f   teorica %.4f   (= p_gana/p_pierde)"
              % (est["razon"], t_razon))
        print("     dif = %+.4f  (EE %.4f)  IC95 [%+.4f, %+.4f]\n"
              % (est["diferencia"], est["se_diferencia"],
                 est["ic95_bajo_diferencia"], est["ic95_alto_diferencia"]))

    print("=" * 74)
    print("ESTIMADOR APLICADO A LOS 8 ESCENARIOS")
    print("=" * 74)
    tabla = tabla_disposition()
    if len(tabla):
        print(tabla.to_string(index=False, float_format=lambda x: "%.4f" % x))
        tabla.to_csv(os.path.join(CARPETA_RESULTADOS, "tabla_disposition.csv"), index=False)
    else:
        print("No hay conteos guardados. Corre primero: python escenarios.py")
