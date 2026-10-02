import os
from pathlib import Path

# Los scripts usan rutas relativas a la raíz del repo (data/, metadata/)
os.chdir(Path(__file__).resolve().parent.parent)
