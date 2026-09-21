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

from escenarios import ESCENARIOS
from estimadores import (tabla_disposition, conteos_sinteticos,
                          estimar_disposition, valor_teorico)

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


if __name__ == "__main__":
    for f in (figura_disposition, figura_robustez, figura_validacion):
        print("generada:", os.path.basename(f()))
