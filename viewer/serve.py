#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Servidor Local para el Visor Cartográfico del Atlas Solar de Córdoba
====================================================================
Convenio ATLAS II — FCEFyN/UNC & OHMC

Ejecución:
    python viewer/serve.py [--port 8080]
"""

import sys
import os
import socket
import argparse
import webbrowser
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

ROOT_DIR = Path(__file__).resolve().parent.parent


class AtlasSolarHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def do_GET(self):
        # Redirigir raíz a /viewer/index.html
        if self.path in ('/', '/index.html'):
            self.send_response(302)
            self.send_header('Location', '/viewer/index.html')
            self.end_headers()
            return
        return super().do_GET()

    def end_headers(self):
        # Habilitar CORS y evitar caché excesiva durante desarrollo
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def main():
    parser = argparse.ArgumentParser(description="Servidor local para el Visor Atlas Solar.")
    parser.add_argument("--port", type=int, default=8080, help="Puerto HTTP (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="No abrir navegador automáticamente")
    args = parser.parse_args()

    port = args.port
    # Si el puerto está ocupado, buscar el siguiente disponible
    while is_port_in_use(port) and port < 8100:
        port += 1

    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, AtlasSolarHandler)

    local_url = f"http://localhost:{port}/viewer/index.html"
    ip_url = f"http://127.0.0.1:{port}/viewer/index.html"

    print("=" * 80)
    print("☀️  ATLAS SOLAR DE CÓRDOBA — VISOR CARTOGRÁFICO GEOTIFF (2 KM)")
    print("    Convenio ATLAS II — FCEFyN / UNC & OHMC")
    print("=" * 80)
    print(f"\n🌐 Servidor web activo y escuchando en el puerto {port}:")
    print(f"    👉 URL Principal:  {local_url}")
    print(f"    👉 URL Alternativa: {ip_url}")
    print(f"\n📁 Directorio raíz servido: {ROOT_DIR}")
    print("💡 Presione Ctrl+C en esta terminal para detener el servidor cuando termine.")
    print("=" * 80 + "\n")

    if not args.no_browser:
        try:
            webbrowser.open(local_url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Servidor detenido por el usuario.")
        httpd.server_close()


if __name__ == "__main__":
    main()
