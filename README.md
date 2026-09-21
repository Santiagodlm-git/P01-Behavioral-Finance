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
disposition-overconfidence-sim/
├── src/                    # Módulos reutilizables del simulador
│   ├── population.py       # Generación de la población de traders (δ, κ, W, n)
│   ├── prices.py            # Motor de precios (GBM + estructura de factores)
│   ├── costs.py              # Comisión + spread bid-ask
│   ├── decision_rule.py      # Regla de venta (utilidad de punto de referencia)
│   └── estimators.py         # [pendiente] PGR/PLR y regresión de overconfidence
├── scenarios/
│   └── run_all_scenarios.py  # [pendiente] corre los 8 escenarios del proyecto
├── notebooks/                # Notebooks de exploración y análisis
├── results/                  # Tablas de resultados generadas (CSV, etc.)
├── report/                   # Reporte final escrito
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
from src.population import generate_population
from src.prices import generar_precios
from src.decision_rule import decide_venta
from src.costs import calcular_costos_transaccion

pop = generate_population(N=1000, delta_center=0.8, kappa_center=0.0, seed=12345)
df_prices = generar_precios(n_days=500, n_assets=50, seed=2024)
```

## Estado del proyecto

- [x] Paso 1 — Población de traders (`src/population.py`)
- [x] Paso 2 — Motor de precios con estructura de factores (`src/prices.py`)
- [x] Paso 3 — Costos de transacción: comisión + spread (`src/costs.py`)
- [x] Paso 4 — Regla de decisión de venta (`src/decision_rule.py`)
- [x] Paso 5 — Turnover / κ
- [ ] Paso 6 — Correr los 8 escenarios
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
