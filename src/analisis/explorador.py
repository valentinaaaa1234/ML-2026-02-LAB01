"""Data Understanding sobre el corpus estructurado.

Las visualizaciones no son decoración: deben revelar cobertura, sesgos y
problemas de calidad (nulos, JSON inválidos, nombres inconsistentes).

Cada método imprime además los números en consola, para poder citarlos en
el informe sin tener que leerlos desde la imagen.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

# Backend sin ventana: solo se guardan archivos PNG, no se abren ventanas.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (debe ir después de matplotlib.use)
import pandas as pd  # noqa: E402

from src.config import DATA_DIR, DIR_JSON  # noqa: E402

# Campos del contrato JSON del laboratorio.
CAMPOS_CONTRATO = [
    "id_noticia",
    "titulo",
    "fecha_publicacion",
    "fuente",
    "url",
    "resumen",
    "delitos",
    "personas",
    "organizaciones",
    "lugares",
    "objetos",
    "relaciones",
]

# Un solo color por gráfico: hay una sola serie, así que el color no
# codifica identidad y no hace falta leyenda.
COLOR_BARRA = "#2f6f9f"
COLOR_ALERTA = "#b4531f"


class ExploradorDatos:
    """Estadísticas y gráficos mínimos del laboratorio."""

    def __init__(self, dir_json: Path = DIR_JSON, dir_figuras: Path | None = None) -> None:
        self.dir_json = dir_json
        self.dir_figuras = dir_figuras or (DATA_DIR / "figuras")
        self._noticias: list[dict] | None = None

    # ------------------------------------------------------------------
    # Carga de datos
    # ------------------------------------------------------------------

    def cargar(self) -> list[dict]:
        """Lee data/json/*.json una sola vez y lo deja en memoria."""
        if self._noticias is not None:
            return self._noticias

        noticias: list[dict] = []
        invalidos: list[str] = []

        for ruta in sorted(self.dir_json.glob("*.json")):
            if ruta.stem.startswith("ejemplo"):
                continue
            try:
                data = json.loads(ruta.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                invalidos.append(ruta.name)
                continue
            if isinstance(data, dict):
                noticias.append(data)
            else:
                invalidos.append(ruta.name)

        if invalidos:
            print(f"JSON inválidos ({len(invalidos)}): {', '.join(invalidos)}")

        self._noticias = noticias
        return noticias

    # ------------------------------------------------------------------
    # Utilidades de graficado
    # ------------------------------------------------------------------

    def _barras_horizontales(
        self,
        conteo: list[tuple[str, float]],
        titulo: str,
        etiqueta_x: str,
        archivo: str,
        color: str = COLOR_BARRA,
        formato_valor: str = "{:.0f}",
    ) -> Path:
        """Gráfico de barras horizontales ordenado de mayor a menor.

        Se usan barras horizontales porque las categorías son nombres largos
        (organizaciones, lugares) que en vertical quedarían ilegibles. El
        orden descendente permite leer el ranking sin comparar alturas.
        """
        self.dir_figuras.mkdir(parents=True, exist_ok=True)

        etiquetas = [par[0] for par in conteo]
        valores = [par[1] for par in conteo]

        # Se invierte para que el mayor quede arriba (matplotlib dibuja de
        # abajo hacia arriba en el eje y).
        etiquetas = etiquetas[::-1]
        valores = valores[::-1]

        alto = max(2.5, 0.42 * len(etiquetas) + 1.2)
        figura, ejes = plt.subplots(figsize=(9, alto))

        ejes.barh(etiquetas, valores, color=color, height=0.65)

        # Valor al final de cada barra: evita tener que mirar el eje.
        maximo = max(valores) if valores else 1
        for indice, valor in enumerate(valores):
            ejes.text(
                valor + maximo * 0.015,
                indice,
                formato_valor.format(valor),
                va="center",
                fontsize=9,
                color="#444444",
            )

        ejes.set_title(titulo, fontsize=13, pad=12, loc="left")
        ejes.set_xlabel(etiqueta_x, fontsize=10)
        ejes.set_xlim(0, maximo * 1.12)

        # Rejilla discreta solo en el eje de magnitud; sin marco.
        ejes.grid(axis="x", color="#dddddd", linewidth=0.8)
        ejes.set_axisbelow(True)
        for lado in ("top", "right", "left"):
            ejes.spines[lado].set_visible(False)
        ejes.tick_params(axis="y", length=0)

        figura.tight_layout()
        ruta = self.dir_figuras / archivo
        figura.savefig(ruta, dpi=150)
        plt.close(figura)
        print(f"  Gráfico guardado: {ruta}")
        return ruta

    @staticmethod
    def _imprimir_conteo(titulo: str, conteo: list[tuple[str, float]]) -> None:
        print(f"\n{titulo}")
        for nombre, valor in conteo:
            print(f"  {nombre}: {valor}")

    # ------------------------------------------------------------------
    # Gráficos pedidos por la guía
    # ------------------------------------------------------------------

    def noticias_por_fuente(self) -> Path | None:
        """Cuántas noticias aportó cada medio: mide la cobertura del corpus."""
        noticias = self.cargar()
        if not noticias:
            print("Sin noticias que analizar.")
            return None

        serie = pd.Series([n.get("fuente") or "desconocida" for n in noticias])
        conteo = list(serie.value_counts().items())

        self._imprimir_conteo("Noticias por fuente:", conteo)
        return self._barras_horizontales(
            conteo,
            "Noticias procesadas por fuente",
            "Cantidad de noticias",
            "noticias_por_fuente.png",
        )

    def delitos_frecuentes(self, top: int = 10) -> Path | None:
        """Delitos más mencionados: describe de qué trata el corpus."""
        noticias = self.cargar()
        if not noticias:
            print("Sin noticias que analizar.")
            return None

        contador: Counter[str] = Counter()
        for data in noticias:
            # Se normaliza a minúscula porque el LLM alterna mayúsculas.
            for delito in data.get("delitos") or []:
                if delito:
                    contador[str(delito).strip().lower()] += 1

        conteo = contador.most_common(top)
        if not conteo:
            print("No se extrajeron delitos.")
            return None

        self._imprimir_conteo(f"Delitos más frecuentes (top {top}):", conteo)
        return self._barras_horizontales(
            conteo,
            f"Delitos más frecuentes (top {top})",
            "Noticias en que aparece",
            "delitos_frecuentes.png",
        )

    def lugares_frecuentes(self, top: int = 10) -> Path | None:
        """Lugares más mencionados: muestra la concentración territorial."""
        noticias = self.cargar()
        if not noticias:
            print("Sin noticias que analizar.")
            return None

        contador: Counter[str] = Counter()
        for data in noticias:
            for lugar in data.get("lugares") or []:
                if lugar:
                    contador[str(lugar).strip().lower()] += 1

        conteo = contador.most_common(top)
        if not conteo:
            print("No se extrajeron lugares.")
            return None

        self._imprimir_conteo(f"Lugares más mencionados (top {top}):", conteo)
        return self._barras_horizontales(
            conteo,
            f"Lugares más mencionados (top {top})",
            "Noticias en que aparece",
            "lugares_frecuentes.png",
        )

    def campos_faltantes(self) -> Path | None:
        """Porcentaje de campos vacíos: es la medida de calidad del corpus.

        Un campo se cuenta como faltante si no existe, si vale null o si es
        una lista vacía. El contrato del laboratorio admite esos tres casos,
        así que los tres representan información que el LLM no encontró.
        """
        noticias = self.cargar()
        if not noticias:
            print("Sin noticias que analizar.")
            return None

        total = len(noticias)
        porcentajes: list[tuple[str, float]] = []
        for campo in CAMPOS_CONTRATO:
            vacios = 0
            for data in noticias:
                valor = data.get(campo)
                if valor is None or valor == "" or valor == []:
                    vacios += 1
            porcentajes.append((campo, round(100 * vacios / total, 1)))

        porcentajes.sort(key=lambda par: par[1], reverse=True)

        self._imprimir_conteo("Porcentaje de campos vacíos:", porcentajes)
        return self._barras_horizontales(
            porcentajes,
            f"Campos vacíos en el JSON extraído (n={total} noticias)",
            "Porcentaje de noticias con el campo vacío",
            "campos_faltantes.png",
            color=COLOR_ALERTA,
            formato_valor="{:.1f}%",
        )

    def evolucion_temporal(self) -> Path | None:
        """Noticias por mes de publicación: cobertura temporal del corpus."""
        noticias = self.cargar()
        if not noticias:
            print("Sin noticias que analizar.")
            return None

        fechas = pd.to_datetime(
            [n.get("fecha_publicacion") for n in noticias],
            errors="coerce",
        )
        fechas = fechas.dropna()
        if len(fechas) == 0:
            print("Ninguna noticia tiene fecha de publicación válida.")
            return None

        sin_fecha = len(noticias) - len(fechas)
        if sin_fecha:
            print(f"Noticias sin fecha válida: {sin_fecha} de {len(noticias)}")

        serie = pd.Series(1, index=fechas).resample("MS").sum()
        etiquetas = [fecha.strftime("%Y-%m") for fecha in serie.index]
        valores = list(serie.values)

        self.dir_figuras.mkdir(parents=True, exist_ok=True)
        figura, ejes = plt.subplots(figsize=(9, 4))
        ejes.bar(etiquetas, valores, color=COLOR_BARRA, width=0.6)

        for indice, valor in enumerate(valores):
            if valor:
                ejes.text(
                    indice,
                    valor + max(valores) * 0.03,
                    str(int(valor)),
                    ha="center",
                    fontsize=9,
                    color="#444444",
                )

        ejes.set_title("Noticias por mes de publicación", fontsize=13, pad=12, loc="left")
        ejes.set_ylabel("Cantidad de noticias", fontsize=10)
        ejes.set_ylim(0, max(valores) * 1.18)
        ejes.grid(axis="y", color="#dddddd", linewidth=0.8)
        ejes.set_axisbelow(True)
        for lado in ("top", "right"):
            ejes.spines[lado].set_visible(False)
        plt.xticks(rotation=45, ha="right")

        figura.tight_layout()
        ruta = self.dir_figuras / "evolucion_temporal.png"
        figura.savefig(ruta, dpi=150)
        plt.close(figura)
        print(f"  Gráfico guardado: {ruta}")
        return ruta

    # ------------------------------------------------------------------
    # Resumen y orquestación
    # ------------------------------------------------------------------

    def resumen_corpus(self) -> None:
        """Cifras generales y detección de duplicados."""
        noticias = self.cargar()
        if not noticias:
            return

        urls = [n.get("url") for n in noticias if n.get("url")]
        titulos = [str(n.get("titulo") or "").strip().lower() for n in noticias]
        urls_repetidas = [u for u, c in Counter(urls).items() if c > 1]
        titulos_repetidos = [t for t, c in Counter(titulos).items() if c > 1 and t]

        entidades = Counter()
        for data in noticias:
            entidades["personas"] += len(data.get("personas") or [])
            entidades["organizaciones"] += len(data.get("organizaciones") or [])
            entidades["lugares"] += len(data.get("lugares") or [])
            entidades["objetos"] += len(data.get("objetos") or [])
            entidades["relaciones"] += len(data.get("relaciones") or [])

        print("\n== Resumen del corpus ==")
        print(f"  Noticias con JSON válido: {len(noticias)}")
        print(f"  URLs duplicadas: {len(urls_repetidas)}")
        print(f"  Títulos duplicados: {len(titulos_repetidos)}")
        for clave, valor in entidades.items():
            promedio = valor / len(noticias)
            print(f"  {clave}: {valor} en total ({promedio:.1f} por noticia)")

    def ejecutar(self) -> None:
        """Corre todas las visualizaciones pedidas en la guía."""
        noticias = self.cargar()
        if not noticias:
            print(
                "No hay JSON en data/json/. "
                "Ejecute primero: python main.py extraer"
            )
            return

        self.resumen_corpus()
        print()
        self.noticias_por_fuente()
        self.delitos_frecuentes()
        self.lugares_frecuentes()
        self.campos_faltantes()
        self.evolucion_temporal()
        print(f"\nFiguras guardadas en: {self.dir_figuras}")
