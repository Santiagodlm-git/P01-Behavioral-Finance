import numpy as np


def decide_compra(kappa_i, cash_disponible, rng, base_prob=0.05): #Se le pone como base inicial de 0.05, que es la probabilidad de compra base de un inversionista sin sesgo de sobreconfianza (k=0)
    """
    Regla de decision de compra (reentrada al mercado), impulsada por kappa_i
    (overconfidence). NO depende de delta_i en ningun calculo -- independencia
    por construccion respecto al mecanismo de disposition effect.

    kappa_i: array de tamaño N (un valor por trader), 0 a 1.
    cash_disponible: array de tamaño N, efectivo disponible ese dia por trader.
    rng: generador de numeros aleatorios (independiente del usado para delta/ventas).
    base_prob: probabilidad de compra de un trader con kappa=0 (piso minimo).

    Regresa: array booleano de tamaño N, True = compra ese dia.
    """
    kappa_i = np.asarray(kappa_i, dtype=float) #Tranforma los inputs a arrays de numpy 
    cash_disponible = np.asarray(cash_disponible, dtype=float)

    prob_compra = base_prob * (1.0 + 5.0 * kappa_i) #Conforme k crece, al probaboilidad de compra aumenta
    prob_compra = np.clip(prob_compra, 0.0, 1.0) #Limita las probabilidades entre 0 y 1

    tiene_cash = cash_disponible > 0
    tira_moneda = rng.random(size=kappa_i.shape) < prob_compra #Materializa la compra dado un numero aleatorio que se genera del 0-1. 

    return tiene_cash & tira_moneda


def calcular_turnover_barber_odean(valor_comprado_total, valor_vendido_total, valor_promedio_cartera):
    """
    Formula estandar de turnover (Barber & Odean, "Trading is Hazardous to
    Your Wealth", paginas 7 y 9):

        Turnover = (valor comprado + valor vendido) / (2 * valor promedio de cartera)

    Los tres argumentos pueden ser escalares o arrays (uno por trader/periodo).
    Si valor_promedio_cartera <= 0, regresa 0.0 en esa posicion (evita division
    por cero cuando un trader se queda sin cartera).
    """
    valor_comprado_total = np.asarray(valor_comprado_total, dtype=float)
    valor_vendido_total = np.asarray(valor_vendido_total, dtype=float)
    valor_promedio_cartera = np.asarray(valor_promedio_cartera, dtype=float)

    denom = 2.0 * valor_promedio_cartera
    turnover = np.divide(
        valor_comprado_total + valor_vendido_total,
        denom,
        out=np.zeros_like(denom, dtype=float),#Si un inversionista se queda sin cartera, simplemente se le asigna un 0, y no calcula nada.
        where=denom > 0,
    )
    return turnover


if __name__ == "__main__":
    rng = np.random.default_rng(55)

    # --- Chequeo de monotonicidad (mismo tipo de prueba que hicimos en Paso 4) ---
    n = 20000
    kappa_fijo_por_nivel = [0.0, 0.1, 0.3, 0.5, 0.8]
    cash = np.full(n, 1000.0)  # todos con efectivo disponible

    print("--- Graduacion de la tasa de compra segun kappa ---")
    for k in kappa_fijo_por_nivel:
        rng_k = np.random.default_rng(999)  # misma seed para comparar limpio
        kappa_array = np.full(n, k)
        compra = decide_compra(kappa_array, cash, rng_k)
        print(f"kappa={k}: % que compra hoy = {compra.mean()*100:.2f}%")

    print()
    # --- Chequeo de la formula de turnover con numeros de juguete ---
    print("--- Formula de turnover ---")
    print("Comprado=200, Vendido=300, Cartera promedio=1000 -> turnover =",
          calcular_turnover_barber_odean(200, 300, 1000))
    print("(esperado: (200+300)/(2*1000) = 0.25)")