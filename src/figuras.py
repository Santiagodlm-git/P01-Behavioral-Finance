"""
Figuras del reporte.

Genera las imagenes que se pegan en el documento a partir de los resultados
guardados en results/. Ninguna figura es una captura de pantalla: si los
numeros cambian, se vuelve a correr este archivo y las imagenes quedan al dia,
con el mismo codigo y las mismas semillas.

Salida: carpeta report/
"""

import os

import matplotlib
matplotlib.use("Agg")          # backend sin ventana, para poder guardar a archivo
import matplotlib.pyplot as plt
import pandas as pd

from escenarios import ESCENARIOS, CARPETA_RESULTADOS
from estimadores import (tabla_disposition, conteos_sinteticos,
                          estimar_disposition, valor_teorico)
from regresion import tabla_overconfidence

CARPETA_REPORTE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report")

COLOR_TEXTO = "#1f2933"
COLOR_ENCABEZADO = "#e4e7eb"
COLOR_FILA_ALT = "#f5f7fa"
COLOR_RESALTE = "#dce9f7"

NOMBRE_CORTO = {
    1: "Nulo", 2: "Disposition baja", 3: "Disposition alta",
    4: "Turnover bajo", 5: "Turnover alto", 6: "Ambos activos",
    7: "Rebalanceo", 8: "Reversion",
}


def _fmt(x, dec=4, signo=False):
    if pd.isna(x):
        return "—"
    return ("%+." + str(dec) + "f") % x if signo else ("%." + str(dec) + "f") % x


# ---------------------------------------------------------------------------
# Motor de dibujo, compartido por las tres tablas
# ---------------------------------------------------------------------------
def _tabla_png(encabezados, filas, titulo, pie, nombre,
                filas_resaltadas=(), columnas_negritas=(), ancho=15.5,
                columna_texto=1):
    """
    Dibuja una tabla y la guarda como PNG.

    filas_resaltadas: indices (base 0) de los renglones a sombrear.
    columnas_negritas: indices de columnas cuyo texto va en negritas.
    columna_texto: columna que contiene texto y se alinea a la izquierda;
        None cuando todas las columnas son numericas y van centradas.

    El ancho de cada columna se calcula a partir del texto mas largo que
    contiene, encabezado incluido. Sin eso matplotlib reparte el ancho en
    partes iguales y las columnas con texto largo se desbordan sobre la vecina.
    """
    anchos = []
    for j, enc in enumerate(encabezados):
        largo = max([len(enc)] + [len(f[j]) for f in filas])
        anchos.append(largo + 2.5)
    total = sum(anchos)
    anchos = [a / total for a in anchos]

    alto = 0.42 * (len(filas) + 1) + 0.9
    fig, ax = plt.subplots(figsize=(ancho, alto))
    ax.axis("off")
    ax.set_position([0.0, 0.10, 1.0, 0.78])   # deja aire solo para titulo y pie

    tabla = ax.table(cellText=filas, colLabels=encabezados, colWidths=anchos,
                     cellLoc="center", loc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10.5)
    tabla.scale(1, 1.55)

    for (fila, col), celda in tabla.get_celld().items():
        celda.set_edgecolor("#cbd2d9")
        celda.set_linewidth(0.6)
        celda.get_text().set_color(COLOR_TEXTO)
        if fila == 0:
            celda.set_facecolor(COLOR_ENCABEZADO)
            celda.get_text().set_fontweight("bold")
        else:
            if (fila - 1) in filas_resaltadas:
                celda.set_facecolor(COLOR_RESALTE)
            elif fila % 2 == 0:
                celda.set_facecolor(COLOR_FILA_ALT)
            else:
                celda.set_facecolor("white")
            if col == columna_texto:
                celda.get_text().set_ha("left")
                celda._text.set_x(0.04)
            if col in columnas_negritas:
                celda.get_text().set_fontweight("bold")

    ax.set_title(titulo, fontsize=13, fontweight="bold", color=COLOR_TEXTO, pad=10)
    fig.text(0.5, 0.025, pie, ha="center", fontsize=8.5, color="#52606d")

    os.makedirs(CARPETA_REPORTE, exist_ok=True)
    ruta = os.path.join(CARPETA_REPORTE, nombre)
    fig.savefig(ruta, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


# ---------------------------------------------------------------------------
# Figura 1 -- tabla principal del Paso 7
# ---------------------------------------------------------------------------
def figura_disposition():
    est = tabla_disposition().set_index("escenario")
    cfg = {c["num"]: c for c in ESCENARIOS}

    encabezados = ["#", "Escenario", "δ", "κ", "Confound", "PGR", "PLR",
                   "PGR − PLR", "EE", "IC 95%", "PGR/PLR"]
    filas, resaltar = [], []
    for i, num in enumerate(est.index):
        r, c = est.loc[num], cfg[num]
        if num in (2, 7, 8):
            resaltar.append(i)
        filas.append([
            str(num), NOMBRE_CORTO[num], "%.1f" % c["delta"], "%.1f" % c["kappa"],
            c["confound"] or "—",
            _fmt(r["PGR"]), _fmt(r["PLR"]), _fmt(r["dif"], signo=True),
            _fmt(r["se_dif"]),
            "[%s, %s]" % (_fmt(r["ic95_bajo"], signo=True), _fmt(r["ic95_alto"], signo=True)),
            "%.3f" % r["razon"],
        ])

    return _tabla_png(
        encabezados, filas,
        "Estimador de disposition effect: PGR y PLR en los 8 escenarios",
        "1,000 cuentas por escenario · 500 días · 50 activos. Errores estándar por bootstrap "
        "de 1,000 réplicas remuestreando cuentas. Sombreados: misma huella, mecanismos distintos.",
        "tabla_disposition.png", filas_resaltadas=resaltar, columnas_negritas=(7,))


# ---------------------------------------------------------------------------
# Figura 2 -- robustez: excluir las realizaciones parciales
# ---------------------------------------------------------------------------
def figura_robustez():
    est = tabla_disposition().set_index("escenario")
    mecanismo = {2: "Sesgo inyectado δ = 0.3",
                 7: "Rebalanceo, δ = 0",
                 8: "Creencia en reversión, δ = 0"}

    encabezados = ["#", "Mecanismo real", "PGR − PLR", "EE",
                   "Excluyendo parciales", "EE", "¿Sobrevive?"]
    filas = []
    for num in (2, 7, 8):
        r = est.loc[num]
        cae = abs(r["dif_sin_parciales"]) < 0.5 * abs(r["dif"])
        filas.append([
            str(num), mecanismo[num],
            _fmt(r["dif"], signo=True), _fmt(r["se_dif"]),
            _fmt(r["dif_sin_parciales"], signo=True), _fmt(r["se_dif_sin_parciales"]),
            "No, se desploma" if cae else "Sí, no se mueve",
        ])

    return _tabla_png(
        encabezados, filas,
        "Prueba de robustez: PGR − PLR excluyendo las realizaciones parciales",
        "Excluir las ventas parciales descarta el rebalanceo, pero no distingue "
        "un sesgo real de una creencia falsa.",
        "tabla_robustez.png", filas_resaltadas=(1,), columnas_negritas=(4,), ancho=13.0)


# ---------------------------------------------------------------------------
# Figura 3 -- validacion del estimador contra datos de respuesta conocida
# ---------------------------------------------------------------------------
def figura_validacion(casos=((0.06, 0.03), (0.04, 0.04), (0.10, 0.02)),
                       n_ganancias=12, n_perdidas=8, seed=1):
    encabezados = ["p(venta | ganancia)", "p(venta | pérdida)",
                   "PGR estimado", "PGR teórico", "PLR estimado", "PLR teórico",
                   "PGR/PLR estimada", "PGR/PLR teórica"]
    filas = []
    for p_g, p_l in casos:
        datos = conteos_sinteticos(p_gana=p_g, p_pierde=p_l, seed=seed)
        est = estimar_disposition(datos, n_replicas=500, seed=seed)
        t_pgr, t_plr, t_razon = valor_teorico(n_ganancias, n_perdidas, p_g, p_l)
        filas.append(["%.2f" % p_g, "%.2f" % p_l,
                      _fmt(est["PGR"]), _fmt(t_pgr),
                      _fmt(est["PLR"]), _fmt(t_plr),
                      "%.4f" % est["razon"], "%.4f" % t_razon])

    return _tabla_png(
        encabezados, filas,
        "Validación del estimador contra datos de respuesta conocida",
        "400 cuentas · 250 días · 12 ganancias y 8 pérdidas por cartera. Los niveles de PGR y PLR "
        "quedan inflados por condicionar en días de venta; la razón recupera exactamente el cociente inyectado.",
        "tabla_validacion.png", columnas_negritas=(6, 7), ancho=15.0,
        columna_texto=None)


# ---------------------------------------------------------------------------
# Figura 4 -- tabla principal del Paso 8
# ---------------------------------------------------------------------------
def figura_overconfidence():
    est = tabla_overconfidence().set_index("escenario")
    cfg = {c["num"]: c for c in ESCENARIOS}

    encabezados = ["#", "Escenario", "δ", "κ", "β bruto", "t", "β neto", "t",
                   "β neto − β bruto", "EE", "Lectura"]
    filas, resaltar = [], []
    for i, num in enumerate(est.index):
        r, c = est.loc[num], cfg[num]
        if abs(r["t_bruto"]) > 2.0:
            lectura = "Causalidad reversa"
            resaltar.append(i)
        elif r["t_neto"] < -1.9:
            lectura = "Costos detectables"
        else:
            lectura = "Costos no detectables"
        filas.append([
            str(num), NOMBRE_CORTO[num], "%.1f" % c["delta"], "%.1f" % c["kappa"],
            _fmt(r["beta_bruto"], 5, signo=True), "%.2f" % r["t_bruto"],
            _fmt(r["beta_neto"], 5, signo=True), "%.2f" % r["t_neto"],
            _fmt(r["diferencia"], 5, signo=True), _fmt(r["se_diferencia"], 5),
            lectura,
        ])

    return _tabla_png(
        encabezados, filas,
        "Estimador de sobreconfianza: regresión de turnover, bruto contra neto",
        "Errores estándar robustos HC1, una observación por cuenta. Controles: log del tamaño de cartera, "
        "posiciones promedio y beta realizada. Sombreados: escenarios donde la pendiente bruta no es cero.",
        "tabla_overconfidence.png", filas_resaltadas=resaltar, columnas_negritas=(4, 8), ancho=17.0)


# ---------------------------------------------------------------------------
# Figura 5 -- diagnostico: el efecto de composicion
# ---------------------------------------------------------------------------
# Valores ANTES del cambio de anclaje. Provienen de la corrida del commit
# 564f12e, anterior a anclar el tamano de posicion al valor actual de la
# cartera. Se dejan escritos aqui porque el codigo actual ya no los reproduce;
# quien quiera verificarlos puede situarse en ese commit y correr regresion.py.
BETA_BRUTO_ANTES = {1: (-0.00822, -2.99), 4: (-0.00148, -1.21),
                     5: (-0.00215, -1.81), 3: (0.02470, 8.88)}


def figura_composicion():
    est = tabla_overconfidence().set_index("escenario")
    etiqueta = {1: "Nulo (δ=0, κ=0)", 4: "Turnover bajo (κ=0.3)",
                5: "Turnover alto (κ=0.8)", 3: "Disposition alta (δ=0.8)"}

    encabezados = ["#", "Escenario", "β bruto antes", "t", "β bruto después", "t", "Diagnóstico"]
    filas = []
    for num in (1, 4, 5, 3):
        antes, t_antes = BETA_BRUTO_ANTES[num]
        desp, t_desp = est.loc[num, "beta_bruto"], est.loc[num, "t_bruto"]
        diag = "Persiste: es causalidad reversa" if abs(t_desp) > 2 else "Desaparece: era composición"
        filas.append([str(num), etiqueta[num],
                      _fmt(antes, 5, signo=True), "%.2f" % t_antes,
                      _fmt(desp, 5, signo=True), "%.2f" % t_desp, diag])

    return _tabla_png(
        encabezados, filas,
        "Diagnóstico de la pendiente bruta: efecto del anclaje del tamaño de posición",
        "Antes: el monto por posición se anclaba al capital inicial Wᵢ/nᵢ. Después: al valor actual de la cartera. "
        "PGR y PLR no cambian con este ajuste.",
        "tabla_composicion.png", filas_resaltadas=(3,), columnas_negritas=(4,), ancho=14.5)


# ---------------------------------------------------------------------------
# Figura 6 -- Paso 9: donde aparece la causalidad reversa
# ---------------------------------------------------------------------------
def figura_causalidad_reversa():
    det = pd.read_csv(os.path.join(CARPETA_RESULTADOS, "diagnostico_reversa.csv"))
    encabezados = ["#", "Escenario", "¿La regla mira el precio de compra?",
                   "β bruto", "t", "¿Significativo?"]
    filas, resaltar = [], []
    for i, (_, r) in enumerate(det.iterrows()):
        num = int(r["escenario"])
        if r["significativo"] == "si":
            resaltar.append(i)
        filas.append([str(num), NOMBRE_CORTO[num],
                      "Sí" if r["regla_condiciona_en_precio_de_compra"] == "si" else "No",
                      _fmt(r["beta_bruto"], 5, signo=True), "%.2f" % r["t_bruto"],
                      "Sí" if r["significativo"] == "si" else "No"])

    return _tabla_png(
        encabezados, filas,
        "Dónde aparece la causalidad reversa",
        "La pendiente bruta solo es distinta de cero donde la regla de venta condiciona sobre el precio de compra "
        "o sobre el movimiento reciente del precio. El escenario 6 es la excepción y se explica en el texto.",
        "tabla_reversa.png", filas_resaltadas=resaltar, columnas_negritas=(3,), ancho=14.5)


# ---------------------------------------------------------------------------
# Figura 7 -- Paso 10: independencia
# ---------------------------------------------------------------------------
def figura_independencia():
    mec = pd.read_csv(os.path.join(CARPETA_RESULTADOS, "independencia_mecanismo.csv"))
    encabezados = ["Medida", "Cuartil δ bajo", "Cuartil δ alto", "Cambio", "corr con δ"]
    filas = []
    for _, r in mec.iterrows():
        v1, v2 = r["cuartil_delta_bajo"], r["cuartil_delta_alto"]
        fmt = (lambda x: "%,.0f".replace(",", "") % x) if abs(v1) > 1000 else (lambda x: "%.3f" % x)
        filas.append([r["medida"], fmt(v1), fmt(v2),
                      "%+.1f%%" % r["cambio_pct"], "%+.4f" % r["corr_con_delta"]])

    return _tabla_png(
        encabezados, filas,
        "Por dónde mueve δ al turnover (κ fijo para toda la población)",
        "El turnover es un cociente: δ casi no cambia el número de operaciones, reduce el valor rotado "
        "y aumenta levemente el tamaño de la cartera. El efecto neto sobre el cociente es negativo.",
        "tabla_independencia.png", filas_resaltadas=(1,), columnas_negritas=(3,), ancho=14.0,
        columna_texto=0)


if __name__ == "__main__":
    for f in (figura_disposition, figura_robustez, figura_validacion,
              figura_overconfidence, figura_composicion,
              figura_causalidad_reversa, figura_independencia):
        print("generada:", os.path.basename(f()))
