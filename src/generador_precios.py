import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Parametros
n_days = 500
n_assets = 50
initial_price = 100.0
mu_annual = 0.08
sigma_annual = 0.20
sigma_market_annual = 0.15
print(f'Parametros: {n_days} Días, {n_assets} Activos, Precio inicial de los activos {initial_price}, Rentabilidad anual {mu_annual}, Volatilidad anual {sigma_annual}, Volatilidad anual del mercado {sigma_market_annual}')

#Usamos un rendimiento anual del 8% para reflejar los promedios históricos de la renta variable a largo plazo y una volatilidad del 20%
#para simular la incertidumbre bursátil real necesaria para activar los sesgos conductuales.

def generador_de_precios(n_days, n_assets, initial_price,
                          mu_annual, sigma_annual, sigma_market_annual, seed):
    #Generador de precios
    rng = np.random.default_rng(seed)

    #Calculos
    dt = 1/252
    drift = (mu_annual - 0.5 * sigma_annual ** 2) * dt
    diffusion = sigma_annual * np.sqrt(dt)

    #Estructura de factores
    market_shock = rng.standard_normal(size=n_days) #Este es eñ ruido de mercado que afecta a todos los activos
    market_factor = sigma_market_annual * np.sqrt(dt) * market_shock #Este es el retorno de mercado que afecta a todos los activos

    betas = rng.normal(loc=1.0, scale=0.3, size=n_assets) #La sensibiliddad de cada activo al factor de mercado (beta) se genera a partir de una distribución normal con mu de 1 y sigma de 0.3.
    #Esto es para que cada activo tenga su propia sensibilidad al ruido del mercado.

    #Componente idiosincrática (explicar la parte de la varianza del activo que queda explicada por su exposición al mercado)
    var_market_contrib = (betas ** 2) * (sigma_market_annual ** 2)
    var_idio = np.clip(sigma_annual ** 2 - var_market_contrib, a_min=1e-6, a_max=None)
    sigma_idio = np.sqrt(var_idio)

    idio_shocks = rng.standard_normal(size=(n_days, n_assets)) #Ruido propio de cada activo, que es independiente del ruido de mercado y de los demás activos.
    idio_component = sigma_idio[np.newaxis, :] * np.sqrt(dt) * idio_shocks #Retorno diario de cada activo por su propio ruido, sin depender del mercado ni de los demas activpos

    #Retorno de cada activo que si viene del mercado, multiplicado por su sensibilidad al mercado (beta)
    market_component = np.outer(market_factor, betas)

    #Retornos diarios finales (sumando todo lo que mueve cada activo)
    daily_returns = drift + market_component + idio_component

    #Matriz de precios (incluye día 0 con el precio inicial)
    log_price_paths = np.vstack([
        np.zeros(n_assets),
        np.cumsum(daily_returns, axis=0)
    ])
    price_matrix = initial_price * np.exp(log_price_paths)

    df_prices = pd.DataFrame(
        price_matrix,
        columns=[f"asset_{i+1}" for i in range(n_assets)]
    )
    return df_prices

df_prices = generador_de_precios(
    n_days=n_days,
    n_assets=n_assets,
    initial_price=initial_price,
    mu_annual=mu_annual,
    sigma_annual=sigma_annual,
    sigma_market_annual=sigma_market_annual,
    seed=2024
)
print(df_prices.head())
print(df_prices.tail())

plt.plot(df_prices)
plt.show()