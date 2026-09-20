"""Validación del JSON producido por el LLM.

El LLM no es la fuente de verdad: el código debe verificar el esquema.

Se distinguen dos niveles:

- Errores: el JSON no cumple el contrato de datos y no se puede usar.
  Lanzan ValueError y el pipeline descarta esa noticia.
- Advertencias: el JSON es utilizable pero tiene problemas de calidad
  (fechas mal formadas, roles ambiguos, relaciones sin respaldo). No
  detienen el proceso; se acumulan para el Data Understanding, que es
  donde la guía pide reportarlos.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from src.conocimiento.utilidades import slugify

# Formato exigido por el contrato del laboratorio.
PATRON_FECHA = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Roles procesales que la guía pide distinguir. No son equivalentes entre
# sí, y confundirlos es uno de los errores éticos que hay que detectar.
#
# Se guardan como raíces y no como palabras completas, para que calcen
# también el femenino y el plural: "imputad" cubre imputado, imputada e
# imputados. Sin esta precaución el validador marcaba como desconocido
# cualquier rol en femenino, que es un falso positivo del propio control.
ROLES_CONOCIDOS = (
    "victim",
    "imputad",
    "detenid",
    "acusad",
    "condenad",
    "testig",
    "sospechos",
    "profug",
    "agresor",
    "autor",
    "fallecid",
    "herid",
    "lesionad",
    "querellante",
    "denunciante",
    "fiscal",
    "familiar",
)

# Roles de personas que aparecen en la noticia pero no participan del
# hecho: autoridades, policías y voceros que solo declaran. No son un
# error de formato, pero inflan el grafo con relaciones que el texto no
# respalda, así que se marcan aparte.
ROLES_NO_PARTICIPANTES = (
    "seremi",
    "delegad",
    "alcalde",
    "general",
    "coronel",
    "subprefect",
    "prefect",
    "comisari",
    "juez",
    "jueza",
    "ministr",
    "abogad",
    "vocer",
    "perit",
    "experto",
    "experta",
    "informante",
    "funcionari",
)


# Palabras que delatan una descripción usada como si fuera un nombre.
MARCADORES_NOMBRE_GENERICO = (
    "victima",
    "imputad",
    "detenid",
    "sujeto",
    "hombre",
    "mujer",
    "menor",
    "adolescente",
    "desconocid",
    "no_identificad",
    "sin_identificar",
)


def _normalizar(texto) -> str:
    """Forma comparable de un nombre: sin tildes, sin espacios, minúscula."""
    return slugify(str(texto or "").strip().lower())


def _es_nombre_generico(nombre: str) -> bool:
    """Detecta marcadores como 'Víctima (hombre)' o 'Mujer de 56 años'.

    El contrato dice que si un dato no aparece se use null. Cuando el LLM
    rellena el nombre con una descripción, crea una entidad que no existe
    y ensucia el vault con notas de personas inventadas.
    """
    normalizado = _normalizar(nombre)
    return any(marcador in normalizado for marcador in MARCADORES_NOMBRE_GENERICO)


class ValidadorJSON:
    """Comprueba que cada archivo JSON cumpla el contrato de datos."""

    CAMPOS_OBLIGATORIOS = [
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

    CAMPOS_LISTA = [
        "delitos",
        "personas",
        "organizaciones",
        "lugares",
        "objetos",
        "relaciones",
    ]

    CAMPOS_TEXTO_NO_VACIO = ["id_noticia", "titulo", "resumen"]

    def __init__(self, mostrar_advertencias: bool = True) -> None:
        self.mostrar_advertencias = mostrar_advertencias
        # Registro acumulado para el informe de calidad.
        self.advertencias: dict[str, list[str]] = {}
        self.conteo_advertencias: Counter[str] = Counter()
        self.invalidos: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Validación de un archivo
    # ------------------------------------------------------------------

    def validar(self, ruta: str | Path) -> dict:
        """Lee, parsea y valida un JSON. Lanza ValueError si rompe el contrato."""
        archivo = Path(ruta)
        try:
            data = json.loads(archivo.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self.invalidos[archivo.name] = f"JSON inválido: {exc}"
            raise ValueError(f"JSON inválido en {archivo}: {exc}") from exc
        except OSError as exc:
            self.invalidos[archivo.name] = f"No se pudo leer: {exc}"
            raise ValueError(f"No se pudo leer {archivo}: {exc}") from exc

        if not isinstance(data, dict):
            self.invalidos[archivo.name] = "No es un objeto JSON"
            raise ValueError(f"{archivo} no contiene un objeto JSON.")

        faltantes = [campo for campo in self.CAMPOS_OBLIGATORIOS if campo not in data]
        if faltantes:
            self.invalidos[archivo.name] = f"Faltan campos: {faltantes}"
            raise ValueError(f"{archivo}: faltan campos {faltantes}")

        for campo in self.CAMPOS_LISTA:
            if not isinstance(data[campo], list):
                self.invalidos[archivo.name] = f"'{campo}' no es lista"
                raise ValueError(f"{archivo}: '{campo}' debe ser una lista")

        for campo in self.CAMPOS_TEXTO_NO_VACIO:
            if not str(data.get(campo) or "").strip():
                self.invalidos[archivo.name] = f"'{campo}' vacío"
                raise ValueError(f"{archivo}: '{campo}' no puede estar vacío")

        # Hasta aquí el contrato se cumple. Lo que sigue son señales de
        # calidad: se registran pero no invalidan la noticia.
        avisos = self._revisar_calidad(data)
        if avisos:
            identificador = str(data.get("id_noticia") or archivo.stem)
            self.advertencias[identificador] = avisos
            for aviso in avisos:
                self.conteo_advertencias[aviso.split(":")[0]] += 1
            if self.mostrar_advertencias:
                for aviso in avisos:
                    print(f"    Aviso: {aviso}")

        return data

    # ------------------------------------------------------------------
    # Revisiones de calidad
    # ------------------------------------------------------------------

    def _revisar_calidad(self, data: dict) -> list[str]:
        avisos: list[str] = []
        avisos.extend(self._revisar_fecha(data))
        avisos.extend(self._revisar_url(data))
        avisos.extend(self._revisar_personas(data))
        avisos.extend(self._revisar_objetos(data))
        avisos.extend(self._revisar_relaciones(data))
        return avisos

    @staticmethod
    def _revisar_fecha(data: dict) -> list[str]:
        """La fecha debe venir como AAAA-MM-DD; si no, no es comparable."""
        fecha = data.get("fecha_publicacion")
        if fecha in (None, ""):
            return ["fecha ausente: sin fecha_publicacion"]
        if not PATRON_FECHA.match(str(fecha).strip()):
            return [f"fecha mal formada: '{fecha}' no cumple AAAA-MM-DD"]
        return []

    @staticmethod
    def _revisar_url(data: dict) -> list[str]:
        """La URL debe conservarse para poder citar la fuente original."""
        url = str(data.get("url") or "").strip()
        if not url:
            return ["url ausente: no se puede citar la fuente"]
        if not url.startswith(("http://", "https://")):
            return [f"url sospechosa: '{url[:60]}'"]
        return []

    @staticmethod
    def _revisar_personas(data: dict) -> list[str]:
        """Cada persona debe traer nombre y un rol procesal reconocible.

        La guía insiste en que detenido, imputado, acusado, condenado,
        víctima y testigo no son equivalentes. Un rol vacío o ambiguo es un
        riesgo ético, porque deja abierta la interpretación de culpabilidad.
        """
        avisos: list[str] = []
        for indice, persona in enumerate(data.get("personas") or []):
            if not isinstance(persona, dict):
                avisos.append(f"persona mal formada: elemento {indice} no es objeto")
                continue
            nombre = str(persona.get("nombre") or "").strip()
            rol = str(persona.get("rol") or "").strip()
            if not nombre:
                # Con el prompt estricto el modelo deja el nombre en null
                # cuando la noticia no lo entrega. Eso CUMPLE el contrato:
                # se registra como estadistica de cobertura, no como error.
                detalle = f"rol '{rol}'" if rol else "sin rol"
                avisos.append(
                    f"persona sin identificar: participante con {detalle}; "
                    "la noticia no entrega su nombre"
                )
                continue
            if not rol:
                avisos.append(f"persona sin rol: '{nombre}'")
                continue
            rol_normalizado = _normalizar(rol)
            if any(marca in rol_normalizado for marca in ROLES_NO_PARTICIPANTES):
                avisos.append(
                    f"persona no participante: '{nombre}' figura como '{rol}', "
                    "declara en la noticia pero no interviene en el hecho"
                )
            elif not any(raiz in rol_normalizado for raiz in ROLES_CONOCIDOS):
                avisos.append(f"rol no reconocido: '{nombre}' aparece como '{rol}'")

            if _es_nombre_generico(nombre):
                avisos.append(
                    f"nombre no identificado: '{nombre}' es una descripción "
                    "y no un nombre propio; el contrato pedía null"
                )
        return avisos

    @staticmethod
    def _revisar_objetos(data: dict) -> list[str]:
        """Un objeto sin tipo ni nombre no aporta nada al grafo."""
        avisos: list[str] = []
        for indice, objeto in enumerate(data.get("objetos") or []):
            if not isinstance(objeto, dict):
                avisos.append(f"objeto mal formado: elemento {indice} no es objeto")
                continue
            if not (objeto.get("nombre") or objeto.get("tipo")):
                avisos.append(f"objeto sin identificar: elemento {indice}")
            cantidad = objeto.get("cantidad")
            if cantidad is not None and not isinstance(cantidad, (int, float)):
                avisos.append(f"cantidad no numérica: '{cantidad}'")
        return avisos

    def _revisar_relaciones(self, data: dict) -> list[str]:
        """Cada relación debe unir entidades declaradas en la misma noticia.

        Es la comprobación que la guía llama "exigir que cada relación esté
        respaldada por el texto". Si el LLM enlaza un origen o un destino
        que no figura en ninguna de las listas de entidades, esa relación no
        tiene respaldo: o inventó la entidad, o la nombró de otra forma. En
        ambos casos el enlace quedaría roto en el vault de Obsidian.
        """
        avisos: list[str] = []
        entidades = self._entidades_declaradas(data)

        for indice, relacion in enumerate(data.get("relaciones") or []):
            if not isinstance(relacion, dict):
                avisos.append(f"relacion mal formada: elemento {indice} no es objeto")
                continue
            origen = str(relacion.get("origen") or "").strip()
            destino = str(relacion.get("destino") or "").strip()
            tipo = str(relacion.get("tipo") or "").strip()
            if not origen or not destino:
                avisos.append(f"relacion incompleta: elemento {indice} sin origen o destino")
                continue
            if not tipo:
                avisos.append(f"relacion sin tipo: '{origen}' -> '{destino}'")
            for extremo, etiqueta in ((origen, "origen"), (destino, "destino")):
                if _normalizar(extremo) not in entidades:
                    avisos.append(
                        f"relacion sin respaldo: {etiqueta} '{extremo}' "
                        "no aparece entre las entidades de la noticia"
                    )
        return avisos

    @staticmethod
    def _entidades_declaradas(data: dict) -> set[str]:
        """Conjunto normalizado de todas las entidades nombradas en la noticia."""
        entidades: set[str] = set()

        for clave in ("delitos", "organizaciones", "lugares"):
            for valor in data.get(clave) or []:
                if valor:
                    entidades.add(_normalizar(valor))

        for persona in data.get("personas") or []:
            nombre = persona.get("nombre") if isinstance(persona, dict) else persona
            if nombre:
                entidades.add(_normalizar(nombre))

        for objeto in data.get("objetos") or []:
            if isinstance(objeto, dict):
                for clave in ("nombre", "tipo"):
                    if objeto.get(clave):
                        entidades.add(_normalizar(objeto[clave]))
            elif objeto:
                entidades.add(_normalizar(objeto))

        return entidades

    # ------------------------------------------------------------------
    # Reporte sobre todo el corpus
    # ------------------------------------------------------------------

    def validar_directorio(self, directorio: str | Path) -> dict:
        """Valida todos los JSON de una carpeta y devuelve el resumen."""
        carpeta = Path(directorio)
        validos = 0
        rechazados = 0

        for ruta in sorted(carpeta.glob("*.json")):
            if ruta.stem.startswith("ejemplo"):
                continue
            try:
                self.validar(ruta)
                validos += 1
            except ValueError as exc:
                rechazados += 1
                print(f"  RECHAZADO {ruta.name}: {exc}")

        return {"validos": validos, "rechazados": rechazados}

    def imprimir_reporte(self) -> None:
        """Resumen de calidad para citar en el informe."""
        print("\n== Reporte de validación ==")
        print(f"  Archivos que rompen el contrato: {len(self.invalidos)}")
        for nombre, motivo in self.invalidos.items():
            print(f"    {nombre}: {motivo}")

        total_avisos = sum(self.conteo_advertencias.values())
        print(f"  Noticias con advertencias: {len(self.advertencias)}")
        print(f"  Advertencias totales: {total_avisos}")
        for tipo, cantidad in self.conteo_advertencias.most_common():
            print(f"    {tipo}: {cantidad}")
