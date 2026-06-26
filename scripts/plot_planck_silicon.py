import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, c, k

# Wavelengths from 200 nm to 3000 nm (converted to meters)
lambdas_nm = np.linspace(200, 3000, 1000)
lambdas_m = lambdas_nm * 1e-9

# Sun temperature
T = 5778 # Kelvin

# Planck's Law (Spectral Radiance)
def planck(wav, T):
    a = 2.0 * h * c**2
    b = h * c / (wav * k * T)
    intensity = a / ( (wav**5) * (np.exp(b) - 1.0) )
    return intensity

B_lambda = planck(lambdas_m, T)
# Normalize to max 1.0 for easy comparison
B_lambda_norm = B_lambda / np.max(B_lambda)

# Typical Silicon Photodiode Response (approximate curve based on typical specs)
# Peak around 900nm, drops to 0 at ~400nm and ~1100nm
si_response = np.zeros_like(lambdas_nm)
mask = (lambdas_nm >= 300) & (lambdas_nm <= 1150)

# Create a smooth bell-like curve for the Si response
# This is an approximation of the image the user uploaded
def si_curve(x):
    # Left side (300 to 900)
    left_mask = (x >= 300) & (x < 900)
    # Right side (900 to 1150)
    right_mask = (x >= 900) & (x <= 1200)
    
    y = np.zeros_like(x)
    y[left_mask] = np.sin((x[left_mask] - 300) / 600 * np.pi / 2)**2
    y[right_mask] = np.cos((x[right_mask] - 900) / 300 * np.pi / 2)**2
    return y

si_response = si_curve(lambdas_nm)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Plot Planck's curve for the Sun
ax.plot(lambdas_nm, B_lambda_norm, color='#E67E22', linewidth=2.5, label='Emisión Solar (Cuerpo Negro a 5778 K)')
ax.fill_between(lambdas_nm, 0, B_lambda_norm, color='#E67E22', alpha=0.2)

# Plot Silicon Photodiode response
ax.plot(lambdas_nm, si_response, color='#2980B9', linewidth=2.5, label='Respuesta Típica Fotodiodo de Silicio (Davis)')
ax.fill_between(lambdas_nm, 0, si_response, color='#2980B9', alpha=0.4)

# Highlight missed regions
missed_mask = (lambdas_nm > 1150)
ax.fill_between(lambdas_nm[missed_mask], 0, B_lambda_norm[missed_mask], color='red', alpha=0.3, hatch='///', label='Energía Infrarroja No Detectada (Error Espectral)')

# Formatting
ax.set_title("Limitación Espectral del Sensor Davis vs. Espectro Solar Idealizado", fontsize=14, weight='bold', pad=15)
ax.set_xlabel("Longitud de Onda $\lambda$ (nm)", fontsize=12)
ax.set_ylabel("Respuesta Relativa / Intensidad Normalizada", fontsize=12)

ax.set_xlim(200, 3000)
ax.set_ylim(0, 1.05)

# Add UV / Visible / IR spans
ax.axvspan(200, 400, color='purple', alpha=0.1)
ax.text(300, 0.95, 'UV', ha='center', color='purple', weight='bold')

ax.axvspan(400, 700, color='green', alpha=0.1)
ax.text(550, 0.95, 'Visible', ha='center', color='green', weight='bold')

ax.axvspan(700, 3000, color='red', alpha=0.05)
ax.text(1800, 0.95, 'Infrarrojo (IR)', ha='center', color='red', weight='bold')

ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='center right', fontsize=10)

import os
os.makedirs('reports/figures', exist_ok=True)
plt.savefig('reports/figures/figura_espectro_silicio.png', dpi=300, bbox_inches='tight')
print("Plot saved to reports/figures/figura_espectro_silicio.png")


