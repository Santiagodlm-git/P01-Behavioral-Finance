"""
Figuras del reporte.

Genera las imagenes que se pegan en el documento, a partir de los resultados
guardados en results/. La idea es que ninguna figura sea una captura de
pantalla: si los numeros cambian, se vuelve a correr este archivo y las
imagenes quedan al dia, con la misma semilla y el mismo codigo.

Salida: carpeta report/
"""

import os

import matplotlib
matplotlib.use("Agg")          # backend sin ventana, para poder guardar en archivo
import matplotlib.pyplot as plt
import pandas as pd

from escenarios import ESCENARIOS
from estimadores import tabla_disposition

CARPETA_REPORTE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report")

# Paleta sobria: un gris oscuro para el texto, un azul discreto para resaltar.
COLOR_TEXTO = "#1f2933"
COLOR_ENCABEZADO = "#e4e7eb"
COLOR_FILA_ALT = "#f5f7fa"
COLOR_RESALTE = "#dce9f7"

# Nombres cortos en espanol para que la tabla quepa sin truncar.
NOMBRE_CORTO = {
    1: "Nulo", 2: "Disposition baja", 3: "Disposition alta",
    4: "Turnover bajo", 5: "Turnover alto", 6: "Ambos activos",
    7: "Rebalanceo", 8: "Reversion",
}


def _fmt(x, dec=4, signo=False):
    if pd.isna(x):
        return "—"
    return ("%+." + str(dec) + "f") % x if signo else ("%." + str(dec) + "f") % x


def tabla_disposition_png(nombre="tabla_disposition.png", resaltar=(2, 7, 8)):
    """
    Dibuja la tabla del estimador de disposition de los 8 escenarios.

    resaltar: escenarios a sombrear. Por defecto el 2, 7 y 8, que son los tres
    que producen diferencias casi identicas con mecanismos distintos.
    """
    est = tabla_disposition().set_index("escenario")
    cfg = {c["num"]: c for c in ESCENARIOS}

    encabezados = ["#", "Escenario", "δ", "κ", "Confound", "PGR", "PLR",
                   "PGR − PLR", "EE", "IC 95%", "PGR/PLR"]
    filas = []
    for num in est.index:
        r, c = est.loc[num], cfg[num]
        filas.append([
            str(num), NOMBRE_CORTO[num], "%.1f" % c["delta"], "%.1f" % c["kappa"],
            c["confound"] or "—",
            _fmt(r["PGR"]), _fmt(r["PLR"]), _fmt(r["dif"], signo=True),
            _fmt(r["se_dif"]),
            "[%s, %s]" % (_fmt(r["ic95_bajo"], signo=True), _fmt(r["ic95_alto"], signo=True)),
            "%.3f" % r["razon"],
        ])

    # Ancho de cada columna proporcional al texto mas largo que contiene
    # (encabezado incluido). Sin esto matplotlib reparte el ancho en partes
    # iguales y las columnas largas, como el intervalo de confianza, se
    # desbordan sobre la vecina.
    anchos = []
    for j, enc in enumerate(encabezados):
        largo = max([len(enc)] + [len(f[j]) for f in filas])
        anchos.append(largo + 2.5)
    total = sum(anchos)
    anchos = [a / total for a in anchos]

    alto = 0.42 * (len(filas) + 1) + 0.9
    fig, ax = plt.subplots(figsize=(15.5, alto))
    ax.axis("off")
    ax.set_position([0.0, 0.10, 1.0, 0.78])   # deja aire solo para titulo y pie

    tabla = ax.table(cellText=filas, colLabels=encabezados, colWidths=anchos,
                     cellLoc="center", loc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10.5)
    tabla.scale(1, 1.55)

    n_col = len(encabezados)
    for (fila, col), celda in tabla.get_celld().items():
        celda.set_edgecolor("#cbd2d9")
        celda.set_linewidth(0.6)
        celda.get_text().set_color(COLOR_TEXTO)
        if fila == 0:                                   # encabezado
            celda.set_facecolor(COLOR_ENCABEZADO)
            celda.get_text().set_fontweight("bold")
        else:
            num_esc = int(filas[fila - 1][0])
            if num_esc in resaltar:
                celda.set_facecolor(COLOR_RESALTE)
            elif fila % 2 == 0:
                celda.set_facecolor(COLOR_FILA_ALT)
            else:
                celda.set_facecolor("white")
            if col == 1:                                # nombre del escenario
                celda.get_text().set_ha("left")
                celda._text.set_x(0.04)
            if col == 7:                                # la columna del efecto
                celda.get_text().set_fontweight("bold")

    ax.set_title("Estimador de disposition effect: PGR y PLR en los 8 escenarios",
                 fontsize=13, fontweight="bold", color=COLOR_TEXTO, pad=10)
    pie = ("1,000 cuentas por escenario · 500 días · 50 activos. "
           "Errores estándar por bootstrap de 1,000 réplicas remuestreando cuentas. "
           "Sombreados: misma huella, mecanismos distintos.")
    fig.text(0.5, 0.025, pie, ha="center", fontsize=8.5, color="#52606d")

    os.makedirs(CARPETA_REPORTE, exist_ok=True)
    ruta = os.path.join(CARPETA_REPORTE, nombre)
    fig.savefig(ruta, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ruta


if __name__ == "__main__":
    ruta = tabla_disposition_png()
    print("figura generada:", os.path.abspath(ruta))
