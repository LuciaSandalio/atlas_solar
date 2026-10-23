#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 5B - Paso 05: Exportación Oficial de Rásters GeoTIFF (COG) y GeoPackage Definitivo
======================================================================================
Convenio ATLAS II — FCEFyN (UNC) & OHMC — Gobierno de la Provincia de Córdoba
Ejecuta la exportación y organización catalogada de todos los productos ráster 2020–2025.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.exportar_catalogo_geotiff_oficial import main

if __name__ == '__main__':
    main()
