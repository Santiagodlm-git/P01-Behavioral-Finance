import numpy as np
import pandas as pd
import time

from generador_poblacion import generate_population
from generador_precios import generador_de_precios
from decision_venta import utilidad_valor
from turnover import calcular_turnover_barber_odean

COMMISSION_BPS = 10.0
SPREAD_BPS = 5.0
TASA_COSTO_TOTAL = (COMMISSION_BPS + SPREAD_BPS) / 10_000  # 15 bps, como fraccion
# Nota: esta tasa reproduce la misma formula de costs.py (comision + spread),
# pero se aplica aqui inline por eficiencia -- costs.py esta pensado para
# operar sobre DataFrames completos, no sobre transacciones individuales
# dentro de un loop dia por dia.

ALPHA = 0.88
BETA_EXP = 0.88
LAM = 2.25
TEMPERATURA = 0.10
C_KAPPA = 0.02  # churn de sobreconfianza: probabilidad diaria extra de rotar una
                # posicion, CIEGA a si va ganando o perdiendo. kappa=1 -> +2 puntos.
H0 = 0.01  # tasa base de venta diaria de una posicion para un agente sin sesgo
           # (delta=0). Implica una tenencia media de ~100 dias habiles y un
           # turnover base del orden de 2.4x anual, cercano al quintil mas
           # activo de Barber & Odean (~250% anual).
MU_ANNUAL = 0.08
SIGMA_ANNUAL = 0.20
DT = 1 / 252


def construir_cartera_inicial(poblacion, price_day0, rng):
    """
    Dia 0: cada trader invierte el 100% de su capital W_i en n_i activos
    elegidos al azar, en partes iguales (W_i / n_i por posicion).

    Regresa 3 matrices de forma (N, max_n) -- una fila por trader, una
    columna por "espacio" de cartera (-1 / 0.0 = espacio vacio):
      held_asset:  id del activo en ese espacio (-1 si vacio)
      held_price:  precio de compra registrado en ese espacio
      held_shares: acciones compradas en ese espacio
    y un array cash (N,) -- todo en cero, porque se invirtio el 100% de W_i.
    """
    N = len(poblacion["delta"])
    n_assets = len(price_day0)
    max_n = int(poblacion["n"].max())

    held_asset = np.full((N, max_n), -1, dtype=int)
    held_price = np.zeros((N, max_n))
    held_shares = np.zeros((N, max_n))
    cash = np.zeros(N)

    for i in range(N):
        ni = int(poblacion["n"][i])
        activos = rng.choice(n_assets, size=ni, replace=False)
        monto_por_posicion = poblacion["W"][i] / ni
        for slot, asset in enumerate(activos):
            precio = price_day0[asset]
            held_asset[i, slot] = asset
            held_price[i, slot] = precio
            held_shares[i, slot] = monto_por_posicion / precio

    return held_asset, held_price, held_shares, cash


def simular_escenario(poblacion, df_prices, seed_decisiones, verbose=False):
    """
    Corre la simulacion dia por dia para UNA poblacion ya generada y UN
    universo de precios ya generado (Opcion A: mismos precios para los 8
    escenarios). Regresa un dict con:
      - transacciones: DataFrame con cada compra/venta (dia, trader, activo,
        tipo, acciones, precio, costo, ganancia_pct para las ventas)
      - cash_final, held_asset/price/shares finales
      - valor_cartera_diario: array (n_days+1, N) para el denominador de turnover
    """
    price_matrix = df_prices.values  # (n_days+1, n_assets)
    n_days_mas_1, n_assets = price_matrix.shape
    N = len(poblacion["delta"])

    seed_seq = np.random.SeedSequence(seed_decisiones)
    seed_init, seed_venta, seed_churn, seed_compra_activo = seed_seq.spawn(4)
    rng_init = np.random.default_rng(seed_init)
    rng_venta = np.random.default_rng(seed_venta)
    rng_churn = np.random.default_rng(seed_churn)
    rng_compra_activo = np.random.default_rng(seed_compra_activo)

    held_asset, held_price, held_shares, cash = construir_cartera_inicial(
        poblacion, price_matrix[0], rng_init
    )

    delta_col = poblacion["delta"].reshape(-1, 1)  # (N,1), broadcast sobre columnas
    kappa = poblacion["kappa"]
    prob_churn_col = (C_KAPPA * kappa).reshape(-1, 1)  # (N,1)
    W = poblacion["W"]
    n_i = poblacion["n"]

    diffusion_diaria = SIGMA_ANNUAL * np.sqrt(DT)

    reg_dia, reg_trader, reg_activo, reg_tipo = [], [], [], []
    reg_acciones, reg_precio, reg_costo, reg_ganancia_pct = [], [], [], []

    # Contadores de las cuatro cajas del estimador de disposition (Paso 7).
    # Se acumulan por cuenta para poder hacer bootstrap POR CUENTA despues.
    cnt_Gr = np.zeros(N, dtype=np.int64)   # ganancias realizadas
    cnt_Gp = np.zeros(N, dtype=np.int64)   # ganancias en papel (no vendidas)
    cnt_Lr = np.zeros(N, dtype=np.int64)   # perdidas realizadas
    cnt_Lp = np.zeros(N, dtype=np.int64)   # perdidas en papel (no vendidas)
    cnt_empate = np.zeros(N, dtype=np.int64)  # posicion exactamente en su precio de compra

    valor_cartera_diario = np.zeros((n_days_mas_1, N))
    valor_cartera_diario[0] = (held_shares * np.where(held_asset >= 0, held_price, 0)).sum(axis=1) + cash

    t0 = time.time()
    for day in range(1, n_days_mas_1):
        price_today = price_matrix[day]

        ocupado = held_asset >= 0
        asset_idx_seguro = np.where(ocupado, held_asset, 0)
        precio_actual = price_today[asset_idx_seguro]
        precio_compra = held_price

        # --- Decision de VENTA (misma logica del Paso 4, vectorizada) ---
        # La utilidad se evalua sobre el RENDIMIENTO, no sobre dolares por accion:
        # asi ganar 10% pesa igual en una accion de $10 que en una de $1,000.
        base_compra = np.where(ocupado, precio_compra, 1.0)  # evita dividir entre 0
        r_ahora = precio_actual / base_compra - 1
        v_ahora = utilidad_valor(r_ahora, ALPHA, BETA_EXP, LAM)
        v_vender = (1 + delta_col) * v_ahora

        r_sube = precio_actual * np.exp(MU_ANNUAL * DT + diffusion_diaria) / base_compra - 1
        r_baja = precio_actual * np.exp(MU_ANNUAL * DT - diffusion_diaria) / base_compra - 1
        v_sube = utilidad_valor(r_sube, ALPHA, BETA_EXP, LAM)
        v_baja = utilidad_valor(r_baja, ALPHA, BETA_EXP, LAM)
        v_continuar = 0.5 * v_sube + 0.5 * v_baja

        # La utilidad MODULA una tasa base H0, no fija la probabilidad desde cero.
        # Con brecha = 0 la probabilidad es exactamente H0; el maximo es 2*H0.
        prob_vende = 2 * H0 / (1 + np.exp(-(v_vender - v_continuar) / TEMPERATURA))
        sorteo = rng_venta.random(size=prob_vende.shape)
        vende_disposition = sorteo < prob_vende

        # --- Canal de sobreconfianza (kappa): churn ciego al precio de compra ---
        # Es un segundo sorteo INDEPENDIENTE que no mira si la posicion va
        # ganando o perdiendo. Asi kappa mueve la frecuencia de operacion sin
        # contaminar el estimador de disposition.
        sorteo_churn = rng_churn.random(size=prob_vende.shape)
        vende_churn = sorteo_churn < prob_churn_col

        vende = ocupado & (vende_disposition | vende_churn)

        # --- Registro para PGR/PLR (Paso 7) ---
        # En los dias en que una cuenta vende ALGO, se clasifican TODAS sus
        # posiciones vivas contra su precio de compra: vendida o retenida,
        # ganancia o perdida. Los dias sin ventas no aportan a ningun conteo.
        # Va ANTES de vaciar los slots, porque despues la informacion se pierde.
        vendio_hoy = vende.any(axis=1)
        if vendio_hoy.any():
            en_juego = ocupado & vendio_hoy[:, None]
            ganancia = precio_actual > precio_compra
            perdida = precio_actual < precio_compra
            cnt_Gr += (en_juego & ganancia & vende).sum(axis=1)
            cnt_Gp += (en_juego & ganancia & ~vende).sum(axis=1)
            cnt_Lr += (en_juego & perdida & vende).sum(axis=1)
            cnt_Lp += (en_juego & perdida & ~vende).sum(axis=1)
            cnt_empate += (en_juego & ~ganancia & ~perdida).sum(axis=1)

        if vende.any():
            tr_idx, slot_idx = np.where(vende)
            activos_vendidos = held_asset[tr_idx, slot_idx]
            acciones_vendidas = held_shares[tr_idx, slot_idx]
            precio_venta = price_today[activos_vendidos]
            precio_orig = held_price[tr_idx, slot_idx]

            valor_bruto = acciones_vendidas * precio_venta
            costo = valor_bruto * TASA_COSTO_TOTAL
            proceeds_neto = valor_bruto - costo

            np.add.at(cash, tr_idx, proceeds_neto)

            ganancia_pct = (precio_venta - precio_orig) / precio_orig

            reg_dia.extend([day] * len(tr_idx))
            reg_trader.extend(tr_idx.tolist())
            reg_activo.extend(activos_vendidos.tolist())
            reg_tipo.extend(["venta"] * len(tr_idx))
            reg_acciones.extend((-acciones_vendidas).tolist())
            reg_precio.extend(precio_venta.tolist())
            reg_costo.extend(costo.tolist())
            reg_ganancia_pct.extend(ganancia_pct.tolist())

            held_asset[tr_idx, slot_idx] = -1
            held_price[tr_idx, slot_idx] = 0.0
            held_shares[tr_idx, slot_idx] = 0.0

        # --- Decision de COMPRA (Paso 5) ---
        ocupado_post_venta = held_asset >= 0
        num_ocupado = ocupado_post_venta.sum(axis=1)
        tiene_espacio = num_ocupado < n_i

        # Reinversion el MISMO dia: el agente sostiene n_i posiciones. Todo lo
        # que vendio hoy lo vuelve a colocar hoy en activos elegidos al azar
        # que no tenga ya. La compra es mecanica y no depende de kappa (kappa
        # actua en el lado de la venta, arriba). Asi el efectivo no se estaciona
        # y el cash drag deja de ser un confound de la regresion del Paso 8.
        compradores = np.where(tiene_espacio & (cash > 0))[0]

        for i in compradores:
            huecos = int(n_i[i]) - int(num_ocupado[i])
            for _ in range(huecos):
                tenidos = set(held_asset[i][held_asset[i] >= 0].tolist())
                opciones = [a for a in range(n_assets) if a not in tenidos]
                if not opciones:
                    break
                activo_elegido = int(rng_compra_activo.choice(opciones))

                monto_objetivo = W[i] / n_i[i]
                monto_max_por_cash = cash[i] / (1 + TASA_COSTO_TOTAL)
                monto_final = min(monto_objetivo, monto_max_por_cash)
                if monto_final <= 0:
                    break

                precio_hoy = price_today[activo_elegido]
                acciones_compradas = monto_final / precio_hoy
                costo = monto_final * TASA_COSTO_TOTAL

                slot_libre = np.where(held_asset[i] == -1)[0][0]
                held_asset[i, slot_libre] = activo_elegido
                held_price[i, slot_libre] = precio_hoy
                held_shares[i, slot_libre] = acciones_compradas
                cash[i] -= (monto_final + costo)

                reg_dia.append(day)
                reg_trader.append(i)
                reg_activo.append(activo_elegido)
                reg_tipo.append("compra")
                reg_acciones.append(acciones_compradas)
                reg_precio.append(precio_hoy)
                reg_costo.append(costo)
                reg_ganancia_pct.append(np.nan)

        valor_cartera_diario[day] = (
            held_shares * np.where(held_asset >= 0, price_today[np.where(held_asset >= 0, held_asset, 0)], 0)
        ).sum(axis=1) + cash

    transacciones = pd.DataFrame({
        "dia": reg_dia, "trader": reg_trader, "activo": reg_activo, "tipo": reg_tipo,
        "acciones": reg_acciones, "precio": reg_precio, "costo_usd": reg_costo,
        "ganancia_pct": reg_ganancia_pct,
    })

    if verbose:
        print(f"Simulacion completa en {time.time()-t0:.1f}s -- {len(transacciones)} transacciones generadas")

    conteos_pgr_plr = pd.DataFrame({
        "trader": np.arange(N),
        "G_r": cnt_Gr, "G_p": cnt_Gp, "L_r": cnt_Lr, "L_p": cnt_Lp,
        "empates": cnt_empate,
    })

    return {
        "transacciones": transacciones,
        "conteos_pgr_plr": conteos_pgr_plr,
        "held_asset": held_asset, "held_price": held_price, "held_shares": held_shares,
        "cash_final": cash,
        "valor_cartera_diario": valor_cartera_diario,
    }


if __name__ == "__main__":
    print("=== Prueba pequeña (20 traders, 30 dias, 10 activos) ===")
    poblacion_chica = generate_population(N=20, delta_center=0.8, kappa_center=0.3, seed=1,
                                           n_min=3, n_max=8)
    precios_chicos = generador_de_precios(n_days=30, n_assets=10, initial_price=100.0,
                                       mu_annual=0.08, sigma_annual=0.20,
                                       sigma_market_annual=0.15, seed=1)
    resultado_chico = simular_escenario(poblacion_chica, precios_chicos, seed_decisiones=1, verbose=True)
    print(resultado_chico["transacciones"].head(10))
    print()

    print("=== Escala completa (1000 traders, 500 dias, 50 activos) ===")
    poblacion_completa = generate_population(N=1000, delta_center=0.8, kappa_center=0.3, seed=42)
    precios_completos = generador_de_precios(n_days=500, n_assets=50, initial_price=100.0,
                                          mu_annual=0.08, sigma_annual=0.20,
                                          sigma_market_annual=0.15, seed=2024)
    resultado_completo = simular_escenario(poblacion_completa, precios_completos, seed_decisiones=42, verbose=True)
    print(resultado_completo["transacciones"]["tipo"].value_counts())