# Simulador de Disposition Effect y Overconfidence

Proyecto para el curso **Comportamientos en las Finanzas y Toma de Decisiones**.

## Objetivo

Construir un simulador de mercado que genera registros de trading sintéticos, inyecta
dos sesgos conductuales (disposition effect vía δ, overconfidence vía κ) a magnitudes
conocidas de antemano, y luego intenta **recuperar esas magnitudes** usando los
estimadores estándar de la literatura (PGR/PLR para disposition, regresión de
turnover para overconfidence).

El punto no es "descubrir" que estos sesgos existen — ya se sabe. El punto es validar
que los estimadores, aplicados sobre datos donde conocemos el ground truth, realmente
recuperan el parámetro inyectado. Cuando no lo recuperan, el trabajo consiste en
averiguar por qué.

## Estructura del repo

```
P01-Behavioral-Finance/
├── src/
│   ├── generador_poblacion.py    # Paso 1 — población de traders (δ, κ, W, n)
│   ├── generador_precios.py      # Paso 2 — precios (GBM + factor de mercado)
│   ├── costs.py                  # Paso 3 — comisión + spread bid-ask
│   ├── decision_venta.py         # Paso 4 — regla de venta (utilidad de punto de referencia)
│   ├── turnover.py               # Paso 5 — fórmula de turnover (Barber & Odean)
│   ├── simulacion.py             # Paso 6 — motor: corre UN escenario día por día
│   ├── escenarios.py             # Paso 6 — corre los 8 escenarios y guarda las tablas
│   ├── estimadores.py            # Paso 7 — PGR/PLR con bootstrap por cuenta
│   ├── regresion.py              # Paso 8 — regresión de turnover con errores HC1
│   └── figuras.py                # Figuras del reporte (PNG generados por código)
├── notebooks/                    # Exploración y análisis
├── results/                      # CSV generados (fuera de git: son derivados)
├── report/                       # Reporte y figuras (PNG fuera de git)
├── requirements.txt
└── README.md
```

Nota sobre nombres: los módulos de los pasos 1, 2 y 4 se llamaban `population.py`,
`prices.py` y `decision_rule.py` hasta el commit `085966d`, donde se renombraron al
español. No se perdió nada, solo cambiaron de nombre.

## Cómo reproducir todo

```bash
pip install -r requirements.txt
cd src
python escenarios.py     # corre los 8 escenarios -> results/*.csv     (~1.5 min)
python estimadores.py    # Paso 7: PGR/PLR + bootstrap -> results/tabla_disposition.csv
python regresion.py      # Paso 8: regresión bruto/neto -> results/tabla_overconfidence.csv
python figuras.py        # las 5 tablas del reporte -> report/*.png
```

El orden importa: `escenarios.py` produce los CSV que consumen los otros tres.

Los módulos se importan entre sí por nombre simple, así que **hay que trabajar desde
la carpeta `src/`**. Cada archivo se puede correr solo y trae su propia demostración:

```python
from generador_poblacion import generate_population
from generador_precios import generador_de_precios
from simulacion import simular_escenario

poblacion = generate_population(N=1000, delta_center=0.8, kappa_center=0.0, seed=101)
precios = generador_de_precios(n_days=500, n_assets=50, initial_price=100.0,
                               mu_annual=0.08, sigma_annual=0.20,
                               sigma_market_annual=0.15, seed=2024)
resultado = simular_escenario(poblacion, precios, seed_decisiones=501)
```

## Estado del proyecto

- [x] Paso 1 — Población de traders (`generador_poblacion.py`)
- [x] Paso 2 — Precios con factor de mercado (`generador_precios.py`)
- [x] Paso 3 — Costos: comisión + spread (`costs.py`)
- [x] Paso 4 — Regla de venta (`decision_venta.py`)
- [x] Paso 5 — Turnover y canal de κ (`turnover.py`, `simulacion.py`)
- [x] Paso 6 — Los 8 escenarios, confounds incluidos (`simulacion.py`, `escenarios.py`)
- [x] Paso 7 — Estimador PGR/PLR con bootstrap por cuenta (`estimadores.py`)
- [x] Paso 8 — Regresión de turnover con errores HC1 (`regresion.py`)
- [ ] Paso 9 — Diagnóstico de confounds
- [ ] Paso 10 — Verificación de independencia δ/κ

## Parámetros de calibración

| Parámetro | Valor | Dónde |
|---|---|---|
| H₀, tasa base de venta diaria | 0.01 (tenencia media ~100 días) | `simulacion.py` |
| τ, temperatura de la logística | 0.10 | `simulacion.py` |
| C, churn de sobreconfianza | 0.02 por unidad de κ | `simulacion.py` |
| Comisión + spread | 10 bps + 5 bps por operación | `simulacion.py`, `costs.py` |
| Rebalanceo (escenario 7) | banda ±5%, revisión cada 21 días | `simulacion.py` |
| Reversión (escenario 8) | θ = 3, ventana de 21 días | `simulacion.py` |
| Población y mercado | 1,000 traders · 500 días · 50 activos | `escenarios.py` |

## Reproducibilidad

Todos los generadores aleatorios usan `numpy.random.default_rng(seed)` con semillas
explícitas — nunca el generador global (`np.random.seed`). Cada escenario deriva su
semilla maestra en cuatro subsemillas independientes (cartera inicial, sorteo de venta,
churn de sobreconfianza y elección del activo), de modo que cambiar un mecanismo no
desplaza los números aleatorios de los demás.

El universo de precios se genera **una sola vez** con semilla fija y se reutiliza en
los 8 escenarios, para que cualquier diferencia observada entre ellos sea atribuible
únicamente a los parámetros de comportamiento inyectados y no a variación de mercado.

Los CSV de `results/` y los PNG de `report/` están fuera de git porque son derivados:
se regeneran corriendo los cuatro scripts de arriba.

## Ramas

| Rama | Para qué |
|---|---|
| `main` | Integración del equipo |
| `fix/calibracion-simulador` | Calibración, confounds y los estimadores de los pasos 7 y 8 |
| `exp/efectos-composicion` | Experimento que identificó el efecto de composición en la pendiente bruta |

## Referencias

- Odean, T. — *Are Investors Reluctant to Realize Their Losses?*
- Barber, B. and Odean, T. — *Trading is Hazardous to Your Wealth*
- Odean, T. — *Do Investors Trade Too Much?*
- Barberis, N. and Xiong, W. — *What Drives the Disposition Effect?*
- Grinblatt, M. and Han, B. — *Prospect Theory, Mental Accounting, and Momentum*
