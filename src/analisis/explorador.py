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

import matplotlib.colors as colors  # noqa: E402
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

    def delitos_por_lugar(self, top_delitos: int = 8, top_lugares: int = 10) -> Path | None:
        """Cruce delito x lugar: dónde se concentra cada tipo de delito.

        La guía pide "delitos por comuna o región", que es un cruce de dos
        variables y no un ranking. Se usa un mapa de calor porque la
        pregunta es de magnitud sobre una grilla: qué celda concentra más
        noticias. Un gráfico de barras agrupadas con 8x10 combinaciones
        sería ilegible.

        La escala es de un solo tono (claro a oscuro) porque codifica
        cantidad, no identidad: usar varios colores sugeriría categorías
        distintas donde solo hay más o menos noticias.
        """
        noticias = self.cargar()
        if not noticias:
            print("Sin noticias que analizar.")
            return None

        # Delitos y lugares más frecuentes: el resto se deja fuera porque
        # aportaría filas y columnas casi vacías.
        conteo_delitos: Counter[str] = Counter()
        conteo_lugares: Counter[str] = Counter()
        for data in noticias:
            for delito in data.get("delitos") or []:
                if delito:
                    conteo_delitos[str(delito).strip().lower()] += 1
            for lugar in data.get("lugares") or []:
                if lugar:
                    conteo_lugares[str(lugar).strip().lower()] += 1

        delitos = [nombre for nombre, _ in conteo_delitos.most_common(top_delitos)]
        lugares = [nombre for nombre, _ in conteo_lugares.most_common(top_lugares)]
        if not delitos or not lugares:
            print("No hay delitos o lugares suficientes para el cruce.")
            return None

        # Celda = número de noticias donde ese delito y ese lugar aparecen juntos.
        matriz = pd.DataFrame(0, index=delitos, columns=lugares, dtype=int)
        for data in noticias:
            delitos_noticia = {
                str(d).strip().lower() for d in (data.get("delitos") or []) if d
            }
            lugares_noticia = {
                str(l).strip().lower() for l in (data.get("lugares") or []) if l
            }
            for delito in delitos_noticia & set(delitos):
                for lugar in lugares_noticia & set(lugares):
                    matriz.loc[delito, lugar] += 1

        self.dir_figuras.mkdir(parents=True, exist_ok=True)

        # Rampa de un solo tono construida desde el azul de los demás
        # gráficos, para que toda la serie de figuras se lea como un sistema.
        rampa = colors.LinearSegmentedColormap.from_list(
            "azul_lab", ["#f2f6fa", COLOR_BARRA]
        )

        figura, ejes = plt.subplots(
            figsize=(1.0 * len(lugares) + 3.2, 0.52 * len(delitos) + 2.4)
        )
        maximo = int(matriz.values.max()) or 1
        malla = ejes.pcolormesh(
            matriz.values,
            cmap=rampa,
            vmin=0,
            vmax=maximo,
            edgecolors="white",
            linewidth=2,
        )

        ejes.set_xticks([i + 0.5 for i in range(len(lugares))])
        ejes.set_yticks([i + 0.5 for i in range(len(delitos))])
        ejes.set_xticklabels(lugares, rotation=45, ha="right", fontsize=9)
        ejes.set_yticklabels(delitos, fontsize=9)
        ejes.invert_yaxis()

        # Valor dentro de cada celda con carga: el color da la lectura
        # rápida, el número permite citarlo en el informe.
        for fila in range(len(delitos)):
            for columna in range(len(lugares)):
                valor = int(matriz.iloc[fila, columna])
                if not valor:
                    continue
                # Texto claro sobre celdas oscuras para mantener contraste.
                tinta = "#ffffff" if valor > maximo * 0.55 else "#2b2b2b"
                ejes.text(
                    columna + 0.5,
                    fila + 0.5,
                    str(valor),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=tinta,
                )

        ejes.set_title(
            "Delitos por lugar (noticias en que aparecen juntos)",
            fontsize=13,
            pad=12,
            loc="left",
        )
        for lado in ("top", "right", "left", "bottom"):
            ejes.spines[lado].set_visible(False)
        ejes.tick_params(length=0)

        barra = figura.colorbar(malla, ax=ejes, shrink=0.75)
        barra.set_label("Noticias", fontsize=9)
        barra.outline.set_visible(False)

        figura.tight_layout()
        ruta = self.dir_figuras / "delitos_por_lugar.png"
        figura.savefig(ruta, dpi=150)
        plt.close(figura)

        print("\nCruce delito x lugar (celdas con valor):")
        for delito in delitos:
            fila = matriz.loc[delito]
            activos = [f"{lugar}={int(v)}" for lugar, v in fila.items() if v]
            if activos:
                print(f"  {delito}: {', '.join(activos)}")
        print(f"  Gráfico guardado: {ruta}")
        return ruta

    def entidades_por_noticia(self) -> Path | None:
        """Cuántas personas y organizaciones trae cada noticia.

        Se grafica la distribución (cuántas noticias tienen 0, 1, 2...
        entidades) y no una barra por noticia: con 39 noticias el eje
        quedaría ilegible y la pregunta interesante no es qué noticia
        puntual trae más, sino si el LLM extrae entidades de forma pareja.

        Dos paneles en vez de dos series superpuestas: las escalas son
        distintas y superponerlas obligaría a leer dos colores donde basta
        con mirar dos veces.
        """
        noticias = self.cargar()
        if not noticias:
            print("Sin noticias que analizar.")
            return None

        self.dir_figuras.mkdir(parents=True, exist_ok=True)
        figura, paneles = plt.subplots(1, 2, figsize=(11, 4.2))

        for panel, (campo, etiqueta) in zip(
            paneles, (("personas", "Personas"), ("organizaciones", "Organizaciones"))
        ):
            cantidades = [len(n.get(campo) or []) for n in noticias]
            distribucion = Counter(cantidades)
            maximo_entidades = max(cantidades) if cantidades else 0
            ejes_x = list(range(maximo_entidades + 1))
            valores = [distribucion.get(x, 0) for x in ejes_x]

            panel.bar(ejes_x, valores, color=COLOR_BARRA, width=0.68)

            tope = max(valores) if valores else 1
            for x, valor in zip(ejes_x, valores):
                if valor:
                    panel.text(
                        x,
                        valor + tope * 0.04,
                        str(valor),
                        ha="center",
                        fontsize=9,
                        color="#444444",
                    )

            promedio = sum(cantidades) / len(cantidades)
            panel.set_title(
                f"{etiqueta} por noticia (promedio {promedio:.1f})",
                fontsize=12,
                pad=10,
                loc="left",
            )
            panel.set_xlabel(f"{etiqueta} extraídas en la noticia", fontsize=10)
            panel.set_ylabel("Cantidad de noticias", fontsize=10)
            panel.set_ylim(0, tope * 1.18)
            panel.set_xticks(ejes_x)
            panel.grid(axis="y", color="#dddddd", linewidth=0.8)
            panel.set_axisbelow(True)
            for lado in ("top", "right"):
                panel.spines[lado].set_visible(False)

            print(f"\n{etiqueta} por noticia:")
            for x in ejes_x:
                if distribucion.get(x):
                    print(f"  {x} {etiqueta.lower()}: {distribucion[x]} noticias")
            print(f"  promedio: {promedio:.2f}")

        figura.tight_layout()
        ruta = self.dir_figuras / "entidades_por_noticia.png"
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
        self.delitos_por_lugar()
        self.entidades_por_noticia()
        self.campos_faltantes()
        self.evolucion_temporal()
        print(f"\nFiguras guardadas en: {self.dir_figuras}")
