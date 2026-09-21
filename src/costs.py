import numpy as np
import pandas as pd


def calcular_costos_transaccion(trade_shares, price, commission_bps=10.0, spread_bps=5.0):
    """
    Calcula el costo de cada transaccion, separando comision y spread bid-ask.

    trade_shares: DataFrame de acciones operadas por dia y activo
                  (positivo = compra, negativo = venta). Mismo shape/index/
                  columns que 'price'.
    price: DataFrame de precios por dia y activo (ej. el df_prices del Paso 2).
    commission_bps: comision explicita del broker, en basis points sobre el
                     valor en dolares operado (10 bps = 0.10%).
    spread_bps: costo del spread bid-ask, en basis points sobre el valor en
                dolares operado -- costo SEPARADO de la comision, aunque
                ambos se expresen en la misma unidad.

    Regresa un dict con:
      - dollar_volume: valor en USD de cada transaccion (|acciones| * precio)
      - commission_usd, spread_usd, total_cost_usd: costos en USD
    """
    if trade_shares.shape != price.shape:
        raise ValueError("trade_shares y price deben tener la misma forma (mismos dias y activos).")
    if not trade_shares.index.equals(price.index):
        raise ValueError(
            "trade_shares y price tienen la misma forma pero indices distintos "
            "-- pandas alinearia por etiqueta y produciria NaN silenciosos. "
            "Reindexa o resetea el indice antes de llamar a esta funcion."
        )
    if not trade_shares.columns.equals(price.columns):
        raise ValueError("trade_shares y price deben tener las mismas columnas, en el mismo orden.")

    dollar_volume = trade_shares.abs() * price
    commission_usd = dollar_volume * (commission_bps / 10_000)
    spread_usd = dollar_volume * (spread_bps / 10_000)
    total_cost_usd = commission_usd + spread_usd

    return {
        "dollar_volume": dollar_volume,
        "commission_usd": commission_usd,
        "spread_usd": spread_usd,
        "total_cost_usd": total_cost_usd,
    }
        


if __name__ == "__main__":
    # Ejemplo: 5 dias, 3 activos -- acciones operadas y precios de esos mismos dias/activos
    rng = np.random.default_rng(1)
    columnas = [f"asset_{i+1}" for i in range(3)]

    trade_shares = pd.DataFrame(rng.integers(-10, 10, size=(5, 3)), columns=columnas)
    price = pd.DataFrame(rng.uniform(80, 150, size=(5, 3)), columns=columnas).round(2)

    resultado = calcular_costos_transaccion(trade_shares, price, commission_bps=10.0, spread_bps=5.0)

    print("Acciones operadas (con signo):")
    print(trade_shares)
    print("\nPrecio ese día:")
    print(price)
    print("\nValor en USD operado:")
    print(resultado["dollar_volume"].round(2))
    print("\nCosto total en USD (comisión + spread):")
    print(resultado["total_cost_usd"].round(4))

    # Verificación: expresar el costo total también como % (bps) del valor operado,
    # para confirmar que "USD" y "basis points" son consistentes entre sí.
    total_bps_efectivo = (resultado["total_cost_usd"] / resultado["dollar_volume"]) * 10_000
    print("\nCosto total efectivo, en basis points (debe dar ~15 = 10+5 en todos lados):")
    print(total_bps_efectivo.round(2))