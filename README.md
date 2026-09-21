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
recuperan el parámetro inyectado.

## Estructura del repo

```
P01-Behavioral-Finance/
├── src/                          # Módulos del simulador
│   ├── generador_poblacion.py    # Paso 1 — población de traders (δ, κ, W, n)
│   ├── generador_precios.py      # Paso 2 — precios (GBM + estructura de factores)
│   ├── costs.py                  # Paso 3 — comisión + spread bid-ask
│   ├── decision_venta.py         # Paso 4 — regla de venta (utilidad de punto de referencia)
│   ├── turnover.py               # Paso 5 — fórmula de turnover (Barber & Odean)
│   ├── simulacion.py             # Paso 6 — motor: corre UN escenario día por día
│   ├── escenarios.py             # Paso 6 — corredor de los 8 escenarios + tablas
│   └── estimadores.py            # [pendiente] Pasos 7 y 8
├── notebooks/                    # Notebooks de exploración y análisis
├── results/                      # Tablas generadas (CSV, ignoradas por git)
├── report/                       # Reporte final escrito
├── requirements.txt
└── README.md
```

## Cómo correrlo

```bash
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Cada módulo en `src/` es independiente y reutilizable — por ejemplo:

```python
# Los módulos se importan entre sí por nombre simple, así que se trabaja
# desde la carpeta src/.
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

- [x] Paso 1 — Población de traders (`src/population.py`)
- [x] Paso 2 — Motor de precios con estructura de factores (`src/prices.py`)
- [x] Paso 3 — Costos de transacción: comisión + spread (`src/costs.py`)
- [x] Paso 4 — Regla de decisión de venta (`src/decision_rule.py`)
- [x] Paso 5 — Turnover / κ
- [~] Paso 6 — Correr los 8 escenarios (`src/escenarios.py`): escenarios 1-6
      corriendo y guardando resultados; 7 y 8 pendientes del diseño de los confounds
- [ ] Paso 7 — Estimador de disposition effect (PGR/PLR)
- [ ] Paso 8 — Estimador de overconfidence (regresión de turnover)
- [ ] Paso 9 — Diagnóstico de confounds
- [ ] Paso 10 — Verificación de independencia δ/κ

## Reproducibilidad

Todos los generadores aleatorios usan `numpy.random.default_rng(seed)` con seeds
explícitas — nunca el generador global (`np.random.seed`). El universo de precios
se genera **una sola vez** (seed fija) y se reutiliza en los 8 escenarios, para que
cualquier diferencia observada entre escenarios sea atribuible únicamente a los
parámetros de comportamiento inyectados (δ, κ), no a variación de mercado.

## Referencias

- Odean, T. — *Are Investors Reluctant to Realize Their Losses?*
- Barber, B. and Odean, T. — *Trading is Hazardous to Your Wealth*
- Odean, T. — *Do Investors Trade Too Much?*
- Barberis, N. and Xiong, W. — *What Drives the Disposition Effect?*
- Grinblatt, M. and Han, B. — *Prospect Theory, Mental Accounting, and Momentum*
