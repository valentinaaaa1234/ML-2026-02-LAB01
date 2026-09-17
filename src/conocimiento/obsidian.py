"""Persistencia final: red de notas Markdown para Obsidian.

No se usa SQLite, MongoDB ni Neo4j. Cada noticia y cada entidad debe
tener su propia nota, enlazada con [[wiki-links]].
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

from src.config import DIR_JSON, DIR_VAULT
from src.conocimiento.utilidades import enlace_obsidian, slugify
from src.excepciones import EtapaPendienteAlumno


class EscritorObsidian(ABC):
    """Contrato para generar la bóveda a partir de JSON validado."""

    @abstractmethod
    def escribir_noticia(self, data: dict) -> Path:
        """Crea obsidian_vault/Noticias/{id_noticia}.md con frontmatter y enlaces."""

    @abstractmethod
    def escribir_entidades(self, noticias: list[dict]) -> None:
        """Agrega notas de delitos, personas, organizaciones, lugares y objetos."""

    @abstractmethod
    def escribir_indice(self, noticias: list[dict]) -> Path:
        """Crea obsidian_vault/00_Indice.md."""

    @abstractmethod
    def escribir_vault(self, noticias: list[dict]) -> None:
        """Orquesta noticia + entidades + índice."""


class EscritorVaultObsidian(EscritorObsidian):
    """Implementación objetivo del laboratorio.

    Jerarquía generada:
        obsidian_vault/
        ├── 00_Indice.md
        ├── Noticias/
        ├── Delitos/
        ├── Personas/
        ├── Organizaciones/
        ├── Lugares/
        ├── Objetos/
        └── Relaciones/
    """

    # Carpeta de cada categoría y el nombre en singular para el campo "Tipo".
    CATEGORIAS = {
        "Delitos": "Delito",
        "Personas": "Persona",
        "Organizaciones": "Organización",
        "Lugares": "Lugar",
        "Objetos": "Objeto",
    }

    def __init__(self, vault: Path = DIR_VAULT, dir_json: Path = DIR_JSON) -> None:
        self.vault = vault
        self.dir_json = dir_json

    # ------------------------------------------------------------------
    # Nombres y enlaces
    # ------------------------------------------------------------------

    @staticmethod
    def _clave(nombre: str) -> str:
        """Clave canónica de una entidad: sin tildes, sin espacios, minúscula.

        El LLM escribe la misma entidad de formas distintas según el campo
        ("Homicidio" en delitos, "homicidio" en relaciones). Si cada variante
        generara su propio archivo, el grafo quedaría fragmentado y los
        enlaces apuntarían a notas inexistentes. Normalizar a minúscula en un
        único punto garantiza que toda la bóveda use el mismo identificador.
        """
        return slugify(str(nombre).strip().lower())

    @classmethod
    def _enlace(cls, nombre: str) -> str:
        """Wiki-link apuntando al nombre de archivo real de la entidad."""
        return enlace_obsidian(cls._clave(nombre))

    # ------------------------------------------------------------------
    # Nota por noticia
    # ------------------------------------------------------------------

    def escribir_noticia(self, data: dict) -> Path:
        """Crea obsidian_vault/Noticias/{id_noticia}.md con frontmatter y enlaces."""
        carpeta = self.vault / "Noticias"
        carpeta.mkdir(parents=True, exist_ok=True)

        id_noticia = data.get("id_noticia") or "sin_id"
        lineas: list[str] = []

        # Frontmatter YAML: metadatos que Obsidian muestra como propiedades.
        lineas.append("---")
        lineas.append(f"id: {id_noticia}")
        lineas.append(f"fecha_publicacion: {data.get('fecha_publicacion') or 'desconocida'}")
        lineas.append(f"fuente: {data.get('fuente') or 'desconocida'}")
        lineas.append(f"url: {data.get('url') or ''}")
        lineas.append("---")
        lineas.append("")

        lineas.append(f"# {data.get('titulo') or id_noticia}")
        lineas.append("")
        lineas.append("## Resumen")
        lineas.append(data.get("resumen") or "Sin resumen.")
        lineas.append("")

        # Listas simples: cada elemento se convierte en un wiki-link.
        for titulo, clave in (
            ("Delitos", "delitos"),
            ("Organizaciones", "organizaciones"),
            ("Lugares", "lugares"),
        ):
            lineas.append(f"## {titulo}")
            elementos = data.get(clave) or []
            if elementos:
                for nombre in elementos:
                    lineas.append(f"- {self._enlace(nombre)}")
            else:
                lineas.append("- Sin registros.")
            lineas.append("")

        # Personas: además del enlace, se anota el rol observado.
        lineas.append("## Personas")
        personas = data.get("personas") or []
        if personas:
            for persona in personas:
                nombre = persona.get("nombre") if isinstance(persona, dict) else persona
                rol = persona.get("rol") if isinstance(persona, dict) else None
                if not nombre:
                    continue
                sufijo = f" — {rol}" if rol else ""
                lineas.append(f"- {self._enlace(nombre)}{sufijo}")
        else:
            lineas.append("- Sin registros.")
        lineas.append("")

        # Objetos: arma, droga, vehículo, dinero incautado, etc.
        lineas.append("## Objetos")
        objetos = data.get("objetos") or []
        if objetos:
            for objeto in objetos:
                if not isinstance(objeto, dict):
                    continue
                nombre = objeto.get("nombre") or objeto.get("tipo")
                if not nombre:
                    continue
                cantidad = objeto.get("cantidad")
                unidad = objeto.get("unidad")
                detalle = ""
                if cantidad is not None:
                    detalle = f" — {cantidad}{' ' + unidad if unidad else ''}"
                lineas.append(f"- {self._enlace(nombre)}{detalle}")
        else:
            lineas.append("- Sin registros.")
        lineas.append("")

        # Relaciones: origen -- tipo --> destino, con ambos extremos enlazados.
        lineas.append("## Relaciones")
        relaciones = data.get("relaciones") or []
        if relaciones:
            for relacion in relaciones:
                if not isinstance(relacion, dict):
                    continue
                origen = relacion.get("origen")
                destino = relacion.get("destino")
                tipo = relacion.get("tipo") or "RELACIONADO_CON"
                if not origen or not destino:
                    continue
                lineas.append(
                    f"- {self._enlace(origen)} -- {tipo} --> {self._enlace(destino)}"
                )
        else:
            lineas.append("- Sin registros.")
        lineas.append("")

        ruta = carpeta / f"{self._clave(id_noticia)}.md"
        ruta.write_text("\n".join(lineas), encoding="utf-8")
        return ruta

    # ------------------------------------------------------------------
    # Índices en memoria
    # ------------------------------------------------------------------

    def _entidades_de(self, data: dict) -> dict[str, list[tuple[str, str]]]:
        """Entidades de una noticia, agrupadas por categoría.

        Devuelve para cada categoría una lista de pares (clave, nombre
        legible). La clave sirve para el archivo y el enlace; el nombre
        legible se usa como título de la nota.
        """
        encontradas: dict[str, list[tuple[str, str]]] = {
            categoria: [] for categoria in self.CATEGORIAS
        }

        def agregar(categoria: str, nombre) -> None:
            if not nombre:
                return
            nombre = str(nombre).strip()
            if not nombre:
                return
            encontradas[categoria].append((self._clave(nombre), nombre))

        for delito in data.get("delitos") or []:
            agregar("Delitos", delito)
        for organizacion in data.get("organizaciones") or []:
            agregar("Organizaciones", organizacion)
        for lugar in data.get("lugares") or []:
            agregar("Lugares", lugar)
        for persona in data.get("personas") or []:
            nombre = persona.get("nombre") if isinstance(persona, dict) else persona
            agregar("Personas", nombre)
        for objeto in data.get("objetos") or []:
            if isinstance(objeto, dict):
                agregar("Objetos", objeto.get("nombre") or objeto.get("tipo"))
            else:
                agregar("Objetos", objeto)

        return encontradas

    def _construir_indices(self, noticias: list[dict]) -> dict:
        """Arma los índices que permiten escribir las notas de entidades.

        Sin base de datos, el agrupamiento se hace en memoria con
        defaultdict(set): para cada entidad se guarda en qué noticias
        aparece, y para cada noticia qué entidades contiene. Con esas dos
        estructuras se puede responder "qué personas aparecen junto a este
        delito" sin recorrer todo el corpus de nuevo.
        """
        apariciones: dict[str, dict[str, set[str]]] = {
            categoria: defaultdict(set) for categoria in self.CATEGORIAS
        }
        nombres: dict[str, dict[str, str]] = {
            categoria: {} for categoria in self.CATEGORIAS
        }
        por_noticia: dict[str, dict[str, set[str]]] = {}

        for data in noticias:
            id_noticia = data.get("id_noticia")
            if not id_noticia:
                continue
            entidades = self._entidades_de(data)
            por_noticia[id_noticia] = {
                categoria: {clave for clave, _ in pares}
                for categoria, pares in entidades.items()
            }
            for categoria, pares in entidades.items():
                for clave, nombre in pares:
                    apariciones[categoria][clave].add(id_noticia)
                    # Se conserva la primera forma legible vista de la entidad.
                    nombres[categoria].setdefault(clave, nombre)

        return {
            "apariciones": apariciones,
            "nombres": nombres,
            "por_noticia": por_noticia,
        }

    # ------------------------------------------------------------------
    # Notas por entidad
    # ------------------------------------------------------------------

    def escribir_entidades(self, noticias: list[dict]) -> None:
        """Una nota por delito, persona, organización, lugar y objeto."""
        indices = self._construir_indices(noticias)
        apariciones = indices["apariciones"]
        nombres = indices["nombres"]
        por_noticia = indices["por_noticia"]

        for categoria, tipo_singular in self.CATEGORIAS.items():
            carpeta = self.vault / categoria
            carpeta.mkdir(parents=True, exist_ok=True)

            for clave, ids_noticias in apariciones[categoria].items():
                lineas: list[str] = []
                lineas.append(f"# {nombres[categoria][clave]}")
                lineas.append("")
                lineas.append(f"Tipo: {tipo_singular}")
                lineas.append("")

                lineas.append("## Noticias relacionadas")
                for id_noticia in sorted(ids_noticias):
                    lineas.append(f"- {self._enlace(id_noticia)}")
                lineas.append("")

                # Entidades que coaparecen: es lo que forma los grupos del
                # laboratorio, sin entrenar ningún algoritmo de clustering.
                for otra_categoria in self.CATEGORIAS:
                    relacionadas: set[str] = set()
                    for id_noticia in ids_noticias:
                        relacionadas |= por_noticia.get(id_noticia, {}).get(
                            otra_categoria, set()
                        )
                    if otra_categoria == categoria:
                        relacionadas.discard(clave)
                    if not relacionadas:
                        continue
                    lineas.append(f"## {otra_categoria} relacionadas")
                    for otra_clave in sorted(relacionadas):
                        lineas.append(f"- {enlace_obsidian(otra_clave)}")
                    lineas.append("")

                ruta = carpeta / f"{clave}.md"
                ruta.write_text("\n".join(lineas), encoding="utf-8")

        self._escribir_relaciones(noticias)

    def _escribir_relaciones(self, noticias: list[dict]) -> None:
        """Una nota por tipo de relación, con todos sus casos y su origen.

        Agrupar por tipo (INVESTIGADO_POR, OPERA_EN, ...) permite revisar de
        una sola vez si el LLM usó los verbos de forma consistente, que es
        parte del control de calidad que pide la guía.
        """
        carpeta = self.vault / "Relaciones"
        carpeta.mkdir(parents=True, exist_ok=True)

        por_tipo: dict[str, list[str]] = defaultdict(list)
        etiquetas: dict[str, str] = {}

        for data in noticias:
            id_noticia = data.get("id_noticia")
            for relacion in data.get("relaciones") or []:
                if not isinstance(relacion, dict):
                    continue
                origen = relacion.get("origen")
                destino = relacion.get("destino")
                tipo = (relacion.get("tipo") or "RELACIONADO_CON").strip()
                if not origen or not destino:
                    continue
                clave_tipo = self._clave(tipo)
                etiquetas.setdefault(clave_tipo, tipo)
                por_tipo[clave_tipo].append(
                    f"- {self._enlace(origen)} -- {tipo} --> {self._enlace(destino)} "
                    f"({self._enlace(id_noticia)})"
                )

        for clave_tipo, filas in por_tipo.items():
            lineas = [f"# {etiquetas[clave_tipo]}", "", "Tipo: Relación", ""]
            lineas.append(f"## Casos registrados ({len(filas)})")
            lineas.extend(sorted(filas))
            lineas.append("")
            ruta = carpeta / f"{clave_tipo}.md"
            ruta.write_text("\n".join(lineas), encoding="utf-8")

    # ------------------------------------------------------------------
    # Índice general
    # ------------------------------------------------------------------

    def escribir_indice(self, noticias: list[dict]) -> Path:
        """Crea obsidian_vault/00_Indice.md, la puerta de entrada de la bóveda."""
        self.vault.mkdir(parents=True, exist_ok=True)
        indices = self._construir_indices(noticias)
        apariciones = indices["apariciones"]
        nombres = indices["nombres"]

        lineas: list[str] = ["# Índice de la bóveda", ""]
        lineas.append(f"Noticias procesadas: {len(noticias)}")
        for categoria in self.CATEGORIAS:
            lineas.append(f"{categoria} distintos: {len(apariciones[categoria])}")
        lineas.append("")

        lineas.append("## Noticias")
        for data in sorted(noticias, key=lambda d: str(d.get("id_noticia"))):
            id_noticia = data.get("id_noticia")
            if not id_noticia:
                continue
            titulo = data.get("titulo") or ""
            lineas.append(f"- {self._enlace(id_noticia)} — {titulo}")
        lineas.append("")

        # Entidades ordenadas por cantidad de noticias: las más repetidas
        # primero, que son las que concentran las relaciones del grafo.
        for categoria in self.CATEGORIAS:
            if not apariciones[categoria]:
                continue
            lineas.append(f"## {categoria}")
            ordenadas = sorted(
                apariciones[categoria].items(),
                key=lambda par: (-len(par[1]), par[0]),
            )
            for clave, ids_noticias in ordenadas:
                lineas.append(
                    f"- {enlace_obsidian(clave)} "
                    f"({len(ids_noticias)} noticia{'s' if len(ids_noticias) != 1 else ''})"
                )
            lineas.append("")

        ruta = self.vault / "00_Indice.md"
        ruta.write_text("\n".join(lineas), encoding="utf-8")
        return ruta

    # ------------------------------------------------------------------
    # Orquestación
    # ------------------------------------------------------------------

    def escribir_vault(self, noticias: list[dict]) -> None:
        """Escribe la bóveda completa: noticias, entidades e índice."""
        if not noticias:
            noticias = self.cargar_json()

        if not noticias:
            raise EtapaPendienteAlumno(
                modulo="src.conocimiento.obsidian.EscritorVaultObsidian.escribir_vault",
                pista=(
                    "No hay JSON en data/json/. "
                    "Ejecute primero: python main.py extraer"
                ),
            )

        for data in noticias:
            self.escribir_noticia(data)
        self.escribir_entidades(noticias)
        ruta_indice = self.escribir_indice(noticias)

        print(f"  Noticias escritas: {len(noticias)}")
        print(f"  Índice: {ruta_indice}")

    def cargar_json(self) -> list[dict]:
        """Lee data/json/*.json, omitiendo archivos inválidos o de ejemplo.

        El archivo ejemplo_N001.json que trae el repositorio se excluye
        porque reutiliza el identificador N001 y sobrescribiría una noticia
        real del corpus.
        """
        noticias: list[dict] = []
        if not self.dir_json.exists():
            return noticias

        for ruta in sorted(self.dir_json.glob("*.json")):
            if ruta.stem.startswith("ejemplo"):
                continue
            try:
                data = json.loads(ruta.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  JSON inválido, se omite {ruta.name}: {exc}")
                continue
            if isinstance(data, dict) and data.get("id_noticia"):
                noticias.append(data)
            else:
                print(f"  Sin id_noticia, se omite {ruta.name}")
        return noticias
