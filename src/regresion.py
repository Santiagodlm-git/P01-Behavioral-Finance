"""
Paso 8 -- Estimador de sobreconfianza (regresion de turnover).

Para cada cuenta i se estima

    r_i = alpha + beta * Turnover_i + gamma' X_i + e_i

dos veces: una con el rendimiento BRUTO y otra con el NETO. El contraste entre
ambos coeficientes es la prueba completa de Barber & Odean (2000):

    beta bruto ~ 0   ->  operar mas no aporta ni destruye valor antes de costos
    beta neto  < 0   ->  operar mas cuesta dinero

Decisiones declaradas:

  * Horizonte: rendimiento del periodo completo (500 dias), sin anualizar.
    Anualizar es una transformacion monotona que solo reescala beta.
  * El rendimiento es geometrico acumulado, V_T/V_0 - 1, de modo que incorpora
    los efectos de composicion que el enunciado advierte que interactuan con delta.
  * Turnover del periodo, no anualizado, para que la relacion entre el
    coeficiente bruto y el neto se lea en las mismas unidades.
  * Controles X_i: tamano de cartera (en logaritmo), numero promedio de
    posiciones y exposicion al riesgo (beta realizada contra el mercado).
  * Errores estandar ROBUSTOS (HC1), sin clustering. Aqui hay UNA observacion
    por cuenta, asi que no existe estructura de grupo que agrupar. El bootstrap
    por cuenta fue necesario en el Paso 7 porque alli cada cuenta aportaba
    cientos de observaciones dependientes entre si.
  * OLS y HC1 implementados a mano con numpy, para que el modulo no dependa de
    statsmodels. La prueba de validacion contrasta contra statsmodels cuando
    esta disponible.
"""

import math
import os

import numpy as np
import pandas as pd

CARPETA_RESULTADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

CONTROLES = ["log_cartera", "posiciones_promedio", "exposicion_riesgo"]
TASA_COSTO_TOTAL = 0.0015   # 15 bps por operacion (10 comision + 5 spread)


# ---------------------------------------------------------------------------
# Bloque 1 -- OLS con errores estandar robustos (HC1)
# ---------------------------------------------------------------------------
def _phi(z):
    """Funcion de distribucion acumulada normal estandar, sin depender de scipy."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def ols_hc1(y, X, nombres=None):
    """
    Minimos cuadrados con errores estandar robustos a heterocedasticidad.

    El estimador puntual es el de siempre: beta = (X'X)^-1 X'y.

    Lo que cambia es la matriz de varianzas. El OLS clasico supone que todos los
    errores tienen la misma varianza. Aqui eso es falso: las cuentas chicas y
    las grandes no tienen la misma dispersion de rendimientos. El estimador de
    White corrige eso usando el cuadrado de cada residuo observado en lugar de
    una varianza comun:

        V = (X'X)^-1  ( sum_i  e_i^2  x_i x_i' )  (X'X)^-1

    HC1 agrega ademas la correccion de grados de libertad n/(n-k), que es lo que
    reportan por defecto Stata y statsmodels.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    # El "pan" del sandwich es (X'X)^-1 y la "carne" es X' diag(e^2) X.
    # Multiplicar cada renglon de X por su residuo al cuadrado y luego X'X
    # calcula esa carne sin construir la matriz diagonal de n x n.
    carne = (X * (resid ** 2)[:, None]).T @ X
    V = XtX_inv @ carne @ XtX_inv * (n / (n - k))

    se = np.sqrt(np.diag(V))
    t = beta / se
    p = np.array([2.0 * (1.0 - _phi(abs(ti))) for ti in t])

    sce = float(resid @ resid)
    sct = float(((y - y.mean()) ** 2).sum())

    return {
        "nombres": nombres or ["x%d" % j for j in range(k)],
        "beta": beta, "se": se, "t": t, "p": p,
        "n": n, "k": k, "r2": 1.0 - sce / sct if sct > 0 else np.nan,
    }


def resumen_ols(res, decimales=5):
    """Imprime el resultado de una regresion en forma de tabla."""
    lineas = ["  %-24s %12s %12s %9s %9s" % ("variable", "coef", "EE (HC1)", "t", "p")]
    for j, nom in enumerate(res["nombres"]):
        lineas.append("  %-24s %12.*f %12.*f %9.2f %9.3f"
                      % (nom, decimales, res["beta"][j], decimales, res["se"][j],
                         res["t"][j], res["p"][j]))
    lineas.append("  n = %d, R2 = %.4f" % (res["n"], res["r2"]))
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Bloque 2 -- Validacion contra datos con beta conocida
# ---------------------------------------------------------------------------
def datos_sinteticos_regresion(n=1000, beta_verdadera=-0.004, seed=0):
    """
    Genera un conjunto donde el coeficiente verdadero se conoce.

    El error se construye HETEROCEDASTICO a proposito: su varianza crece con el
    turnover. Asi la prueba no solo verifica que el coeficiente se recupere,
    sino que el error estandar robusto sea el adecuado. Si se usaran errores
    clasicos en estos datos, estarian mal calibrados.
    """
    rng = np.random.default_rng(seed)
    turnover = rng.gamma(shape=9.0, scale=1.3, size=n)
    tamano = rng.normal(12.0, 0.8, size=n)
    posiciones = rng.integers(5, 31, size=n).astype(float)
    riesgo = rng.normal(1.0, 0.05, size=n)

    ruido = rng.normal(0, 0.03 * (1 + 0.15 * turnover), size=n)
    y = (0.18 + beta_verdadera * turnover + 0.004 * tamano
         + 0.0008 * posiciones + 0.05 * riesgo + ruido)

    return pd.DataFrame({"y": y, "turnover_periodo": turnover, "log_cartera": tamano,
                          "posiciones_promedio": posiciones, "exposicion_riesgo": riesgo})


def matriz_diseno(datos, controles=CONTROLES):
    """Arma X con la constante, el turnover y los controles, en ese orden."""
    columnas = [np.ones(len(datos)), datos["turnover_periodo"].to_numpy(dtype=float)]
    nombres = ["constante", "turnover"]
    for c in controles:
        columnas.append(datos[c].to_numpy(dtype=float))
        nombres.append(c)
    return np.column_stack(columnas), nombres


# ---------------------------------------------------------------------------
# Bloque 3 -- La regresion sobre los escenarios
# ---------------------------------------------------------------------------
def cargar_escenario(num):
    """Lee el resumen por cuenta y agrega el logaritmo del tamano de cartera."""
    ruta = os.path.join(CARPETA_RESULTADOS, "resumen_cuentas_esc%d.csv" % num)
    datos = pd.read_csv(ruta)
    datos["log_cartera"] = np.log(datos["valor_promedio_cartera"].clip(lower=1.0))
    return datos


def regresion_rendimiento(datos, columna, controles=CONTROLES):
    """Corre la regresion para una de las dos definiciones de rendimiento."""
    X, nombres = matriz_diseno(datos, controles)
    return ols_hc1(datos[columna].to_numpy(dtype=float), X, nombres)


def tabla_overconfidence(numeros=range(1, 9), controles=CONTROLES):
    """
    Corre las dos regresiones en cada escenario y arma la tabla de resultados.
    """
    filas = []
    for num in numeros:
        ruta = os.path.join(CARPETA_RESULTADOS, "resumen_cuentas_esc%d.csv" % num)
        if not os.path.exists(ruta):
            continue
        datos = cargar_escenario(num)

        bruto = regresion_rendimiento(datos, "retorno_bruto", controles)
        neto = regresion_rendimiento(datos, "retorno_neto", controles)
        j = 1   # posicion del turnover en el vector de coeficientes

        filas.append({
            "escenario": num,
            "beta_bruto": bruto["beta"][j], "se_bruto": bruto["se"][j],
            "t_bruto": bruto["t"][j], "p_bruto": bruto["p"][j],
            "beta_neto": neto["beta"][j], "se_neto": neto["se"][j],
            "t_neto": neto["t"][j], "p_neto": neto["p"][j],
            "diferencia": neto["beta"][j] - bruto["beta"][j],
            "turnover_medio": datos["turnover_periodo"].mean(),
            "desv_turnover": datos["turnover_periodo"].std(),
            "r2_neto": neto["r2"], "n": neto["n"],
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Bloque 4 -- Diagnostico del coeficiente bruto
# ---------------------------------------------------------------------------
def diagnostico_beta_bruto(num):
    """
    Estima el coeficiente bruto bajo especificaciones anidadas para ver de donde
    proviene, si es que no es cero.

    El enunciado enumera cuatro causas no conductuales de una pendiente bruta
    negativa. La fuga de informacion esta descartada por diseno y el spread
    tambien, porque los costos se registran fuera de la trayectoria de precios.
    Quedan el cash drag y los efectos de composicion, mas la causalidad reversa,
    que empuja en sentido contrario. Esta funcion los separa agregando controles
    de uno en uno.
    """
    datos = cargar_escenario(num)
    especificaciones = [
        ("sin controles", []),
        ("+ tamano de cartera", ["log_cartera"]),
        ("+ posiciones", ["log_cartera", "posiciones_promedio"]),
        ("+ riesgo (X_i completo)", CONTROLES),
        ("+ efectivo", CONTROLES + ["frac_efectivo_promedio"]),
    ]
    filas = []
    for etiqueta, controles in especificaciones:
        res = regresion_rendimiento(datos, "retorno_bruto", controles)
        filas.append({"especificacion": etiqueta, "beta_bruto": res["beta"][1],
                       "se": res["se"][1], "t": res["t"][1], "r2": res["r2"]})
    return pd.DataFrame(filas)


if __name__ == "__main__":
    print("=" * 78)
    print("VALIDACION DE LA REGRESION CONTRA DATOS DE BETA CONOCIDA")
    print("=" * 78)
    for b_verdadera in (-0.004, 0.0, 0.002):
        datos = datos_sinteticos_regresion(beta_verdadera=b_verdadera, seed=7)
        X, nombres = matriz_diseno(datos)
        res = ols_hc1(datos["y"].to_numpy(), X, nombres)
        b, ee = res["beta"][1], res["se"][1]
        dentro = abs(b - b_verdadera) < 1.96 * ee
        print("  beta verdadera %+.4f -> estimada %+.6f (EE %.6f)  %s"
              % (b_verdadera, b, ee, "dentro del IC95" if dentro else "FUERA del IC95"))

    try:
        import statsmodels.api as sm
        datos = datos_sinteticos_regresion(beta_verdadera=-0.004, seed=7)
        X, _ = matriz_diseno(datos)
        ref = sm.OLS(datos["y"].to_numpy(), X).fit(cov_type="HC1")
        mio = ols_hc1(datos["y"].to_numpy(), X)
        print("\n  Contraste contra statsmodels (HC1):")
        print("     coeficiente: %.10f vs %.10f" % (mio["beta"][1], ref.params[1]))
        print("     error est. : %.10f vs %.10f" % (mio["se"][1], ref.bse[1]))
        print("     diferencia maxima en todos los coeficientes: %.2e"
              % np.abs(mio["se"] - ref.bse).max())
    except ImportError:
        print("\n  (statsmodels no instalado: se omite el contraste)")

    print()
    print("=" * 78)
    print("REGRESION DE TURNOVER EN LOS 8 ESCENARIOS")
    print("=" * 78)
    tabla = tabla_overconfidence()
    if len(tabla):
        vista = tabla[["escenario", "beta_bruto", "se_bruto", "t_bruto",
                       "beta_neto", "se_neto", "t_neto", "diferencia"]]
        print(vista.to_string(index=False, float_format=lambda x: "%.5f" % x))
        tabla.to_csv(os.path.join(CARPETA_RESULTADOS, "tabla_overconfidence.csv"), index=False)

        print("\n  Recordatorio del pre-analisis: se espera beta bruto ~ 0,")
        print("  beta neto < 0, y diferencia ~ -0.0030 en los ocho escenarios.")
    else:
        print("No hay resumenes por cuenta. Corre primero: python escenarios.py")
