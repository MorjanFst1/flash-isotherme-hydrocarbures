# ============================================================
# FLASH ISOTHERME : PROPANE / n-BUTANE / n-PENTANE / n-HEXANE
# Méthode : équation de Rachford-Rice
# Modèle d'équilibre : loi de Raoult + équation d'Antoine
# ============================================================

import numpy as np
from scipy.optimize import brentq


# 1. DONNÉES DU PROBLÈME
F = 100.0  # Débit molaire total (kmol/h)

z = {
    "propane": 0.10,
    "n-butane": 0.20,
    "n-pentane": 0.30,
    "n-hexane": 0.40,
}

composants = list(z.keys())
z_vec = np.array([z[c] for c in composants], dtype=float)

if not np.isclose(z_vec.sum(), 1.0, atol=1e-9):
    raise ValueError("La somme des fractions molaires z_i doit être égale à 1.")

T_F = 200.0
T_C = (T_F - 32.0) * 5.0 / 9.0

P_psia = 100.0
P_atm = P_psia / 14.6959
P_mmHg = P_atm * 760.0
P_kPa = P_atm * 101.325


# 2. CONSTANTES D'ANTOINE
# Forme utilisée : log10(Psat en mmHg) = A - B / (C + T en °C)
# Vérifier le domaine de validité des constantes avant usage industriel.
ANTOINE = {
    "propane":   {"A": 6.82973, "B": 813.20,  "C": 248.000},
    "n-butane":  {"A": 6.80896, "B": 935.86,  "C": 238.730},
    "n-pentane": {"A": 6.85221, "B": 1064.63, "C": 232.000},
    "n-hexane":  {"A": 6.87776, "B": 1171.53, "C": 224.366},
}


def psat(nom, temperature_c):
    """Retourne la pression de vapeur saturante en mmHg."""
    p = ANTOINE[nom]
    return 10 ** (p["A"] - p["B"] / (p["C"] + temperature_c))


# 3. COEFFICIENTS DE PARTAGE K_i
Psat_mmHg = np.array([psat(c, T_C) for c in composants])
K = Psat_mmHg / P_mmHg


# 4. ÉQUATION DE RACHFORD-RICE
def rachford_rice(beta, z_values, k_values):
    """Fonction de Rachford-Rice dont la racine donne la fraction vaporisée."""
    return np.sum(z_values * (k_values - 1.0) /
                  (1.0 + beta * (k_values - 1.0)))


f0 = rachford_rice(0.0, z_vec, K)
f1 = rachford_rice(1.0, z_vec, K)

# Détermination de l'état du mélange.
if np.isclose(f0, 0.0, atol=1e-12):
    beta = 0.0
    etat = "À la limite de l'apparition de vapeur (liquide saturé)"
elif np.isclose(f1, 0.0, atol=1e-12):
    beta = 1.0
    etat = "À la limite de l'apparition de liquide (vapeur saturée)"
elif f0 > 0.0 and f1 < 0.0:
    beta = brentq(rachford_rice, 0.0, 1.0, args=(z_vec, K))
    etat = "Deux phases : liquide + vapeur"
elif f0 < 0.0:
    beta = 0.0
    etat = "Tout liquide selon le test Rachford-Rice"
else:
    beta = 1.0
    etat = "Tout vapeur selon le test Rachford-Rice"


# 5. COMPOSITIONS ET DÉBITS
if 0.0 < beta < 1.0:
    x_vec = z_vec / (1.0 + beta * (K - 1.0))
    y_vec = K * x_vec
elif beta == 0.0:
    # En l'absence de vapeur, la composition du liquide est z.
    x_vec = z_vec.copy()
    y_vec = np.full_like(z_vec, np.nan)  # Phase vapeur absente
else:
    # En l'absence de liquide, la composition de vapeur est z.
    y_vec = z_vec.copy()
    x_vec = np.full_like(z_vec, np.nan)  # Phase liquide absente

V = beta * F
L = F - V


# 6. AFFICHAGE DES RÉSULTATS
print("=" * 72)
print("FLASH ISOTHERME — MÉLANGE D'HYDROCARBURES LÉGERS")
print("=" * 72)
print(f"Température : {T_F:.2f} °F = {T_C:.3f} °C")
print(f"Pression    : {P_psia:.2f} psia = {P_kPa:.3f} kPa = {P_atm:.4f} atm")
print(f"Débit total : {F:.3f} kmol/h")
print(f"État calculé : {etat}")
print("-" * 72)

print("Pressions de vapeur saturante et coefficients K_i :")
for c, p_sat, k in zip(composants, Psat_mmHg, K):
    print(f"  {c:12s} Psat = {p_sat:10.3f} mmHg ; K = {k:10.5f}")

print("-" * 72)
print(f"Rachford-Rice : f(0) = {f0:+.6f} ; f(1) = {f1:+.6f}")
print(f"Fraction vaporisée beta = {beta:.6f}")
print(f"Débit liquide L = {L:.4f} kmol/h")
print(f"Débit vapeur  V = {V:.4f} kmol/h")
print("-" * 72)

print("Composition globale z_i :")
for c, zi in zip(composants, z_vec):
    print(f"  z_{c:10s} = {zi:.6f}")

print("-" * 72)
print("Composition liquide x_i :")
for c, xi in zip(composants, x_vec):
    print(f"  x_{c:10s} = {xi:.6f}" if np.isfinite(xi)
          else f"  x_{c:10s} = phase absente")
if np.all(np.isfinite(x_vec)):
    print(f"  Somme x_i = {x_vec.sum():.6f}")

print("-" * 72)
print("Composition vapeur y_i :")
for c, yi in zip(composants, y_vec):
    print(f"  y_{c:10s} = {yi:.6f}" if np.isfinite(yi)
          else f"  y_{c:10s} = phase absente")
if np.all(np.isfinite(y_vec)):
    print(f"  Somme y_i = {y_vec.sum():.6f}")

print("-" * 72)
print("Vérification du bilan de matière par composant :")
if 0.0 < beta < 1.0:
    for c, zi, xi, yi in zip(composants, z_vec, x_vec, y_vec):
        debit_sortie = L * xi + V * yi
        debit_entree = F * zi
        print(f"  {c:12s} sortie = {debit_sortie:10.4f} ; "
              f"entrée = {debit_entree:10.4f} kmol/h")
else:
    print("  Une seule phase est prédite ; le bilan global reste F = L + V.")
    print(f"  F = {F:.4f} ; L + V = {L + V:.4f} kmol/h")

print("=" * 72)
print("REMARQUE : modèle idéal (loi de Raoult) avec constantes d'Antoine.")
print("Pour une étude industrielle, vérifier les constantes et comparer avec")
print("une équation d'état telle que Peng-Robinson ou SRK.")
