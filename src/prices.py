import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generar_precios(n_days=500, n_assets=50, initial_price=100.0,
                     mu_annual=0.08, sigma_annual=0.20, seed=None):
    """
    Simula precios diarios para n_assets activos usando GBM (retornos log-normales).

    Se usa UNA SOLA VEZ (Opción A): el mismo universo de precios se reutiliza
    en los 8 escenarios, para que cualquier diferencia entre escenarios sea
    atribuible solo a los parámetros de comportamiento (delta, kappa), no a
    variación de mercado.

    seed: obligatorio pasarla explícitamente (reproducibilidad, ver Paso 1).
    Regresa un DataFrame de tamaño (n_days+1) x n_assets (incluye precio inicial).
    """
    rng = np.random.default_rng(seed)

    dt = 1 / 252
    drift = (mu_annual - 0.5 * sigma_annual ** 2) * dt
    diffusion = sigma_annual * np.sqrt(dt)

    random_shocks = rng.standard_normal(size=(n_days, n_assets))
    daily_log_returns = drift + diffusion * random_shocks

    # Precio en t=0 = initial_price para todos los activos, luego se acumulan los retornos.
    log_price_paths = np.vstack([
        np.zeros(n_assets),
        np.cumsum(daily_log_returns, axis=0)
    ])
    price_matrix = initial_price * np.exp(log_price_paths)

    df_prices = pd.DataFrame(
        price_matrix,
        columns=[f"asset_{i+1}" for i in range(n_assets)]
    )
    df_prices.index.name = "day"
    return df_prices


def plot_precios(df_prices, highlight_asset="asset_1"):
    """Grafica todos los activos en gris tenue y resalta uno para referencia visual."""
    plt.figure(figsize=(12, 6))
    for col in df_prices.columns:
        plt.plot(df_prices.index, df_prices[col], color="gray", alpha=0.3, linewidth=0.8)
    plt.plot(df_prices.index, df_prices[highlight_asset], color="#1f77b4",
              linewidth=1.8, label=highlight_asset)
    plt.title("Simulación de precios (GBM)", fontsize=12, fontweight="bold")
    plt.xlabel("Días de operación")
    plt.ylabel("Precio ($)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Generación única, con seed fija -- este mismo df_prices se reutiliza
    # en los 8 escenarios (Opción A).
    df_prices = generar_precios(seed=2024)
    print(df_prices.head())
    print(df_prices.tail())
    plot_precios(df_prices)
