"""Extractor Gemini: texto limpio → JSON del contrato del laboratorio.

El laboratorio NO inventa datos: solo se extrae información explícita en
la noticia, en JSON válido.

Mejoras implementadas sobre la versión base (TODO(alumno) del enunciado):
- Reintentos con espera creciente ante errores transitorios (429, 503, timeouts).
- Recorte de textos muy largos antes de armar el prompt.
- Reutilización de los JSON ya generados, para no gastar cuota dos veces.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path

from src.config import DIR_JSON, GEMINI_API_KEY, GEMINI_MODEL, PAUSA_ENTRE_REQUESTS
from src.modelos import NoticiaFuente


class ExtractorLLM(ABC):
    """Interfaz de cualquier extractor basado en modelo generativo."""

    @abstractmethod
    def construir_prompt(self, noticia: NoticiaFuente) -> str:
        """Arma el prompt con el esquema JSON y el texto de la noticia."""

    @abstractmethod
    def extraer(self, noticia: NoticiaFuente) -> dict:
        """Devuelve un diccionario que cumple el contrato JSON del laboratorio."""


class ExtractorGemini(ExtractorLLM):
    """Extractor oficial del laboratorio (Gemini).

    1. Carga GEMINI_API_KEY desde .env (nunca hardcodear la clave).
    2. Usa el texto ya limpio en noticia.texto_limpio.
    3. Llama al modelo indicado en GEMINI_MODEL con construir_prompt().
    4. Parsea JSON (quita fences markdown si el modelo los agrega).
    5. Guarda data/json/{id_noticia}.json.
    """

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

    # Reintentos ante errores transitorios de la API.
    INTENTOS_MAXIMOS = 5
    ESPERA_INICIAL = 4.0  # segundos; se duplica en cada intento fallido
    # Marcas de cuota diaria agotada: reintentar no sirve, hay que esperar
    # al reinicio diario o cambiar de modelo.
    ERRORES_CUOTA_DIARIA = (
        "PerDay",
        "per day",
        "free_tier_requests",
        "GenerateRequestsPerDayPerProjectPerModel",
    )
    ERRORES_TRANSITORIOS = (
        "429",
        "503",
        "500",
        "UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
        "DEADLINE_EXCEEDED",
        "INTERNAL",
        "timeout",
        "overloaded",
        "high demand",
    )

    # Recorte defensivo: los textos muy largos encarecen y degradan la extracción.
    MAX_CARACTERES_TEXTO = 12000

    # Vocabularios cerrados del prompt. Sin ellos el modelo copia la
    # formulación literal de cada noticia y genera decenas de variantes
    # para la misma idea, lo que fragmenta el grafo de Obsidian.
    ROLES_PERMITIDOS = (
        "victima, imputado, detenido, acusado, condenado, testigo, profugo"
    )
    DELITOS_PERMITIDOS = (
        "homicidio, femicidio, parricidio, homicidio frustrado, "
        "robo con violencia, robo con homicidio, robo, hurto, "
        "trafico de drogas, microtrafico, porte ilegal de armas, "
        "lesiones, secuestro, amenazas, receptacion, otro"
    )
    RELACIONES_PERMITIDAS = (
        "VICTIMA_DE, IMPUTADO_POR, DETENIDO_EN, OCURRIO_EN, INVESTIGA, "
        "OPERA_EN, UTILIZO, INCAUTADO_EN, PERTENECE_A"
    )

    def __init__(self, dir_json: Path = DIR_JSON, reusar_existentes: bool = True) -> None:
        self.dir_json = dir_json
        self.dir_json.mkdir(parents=True, exist_ok=True)
        self.reusar_existentes = reusar_existentes
        self._cliente = None

    def construir_prompt(self, noticia: NoticiaFuente) -> str:
        """Prompt con esquema estricto y vocabularios cerrados.

        La versión inicial solo pedía "información explícita" y dejaba los
        valores libres. Medido con el validador sobre 14 noticias, eso
        producía tres problemas: el 44% de las personas extraídas eran
        descripciones ("Hombre de 29 años") en vez de nombres, había
        autoridades que solo declaran registradas como participantes del
        hecho, y relaciones cuyos extremos no coincidían con ninguna
        entidad declarada, lo que rompía los enlaces del vault.

        Este prompt cierra esos tres huecos con reglas explícitas y
        vocabularios acotados.
        """
        campos = ", ".join(self.CAMPOS_OBLIGATORIOS)
        texto = (noticia.texto_limpio or "").strip()
        if len(texto) > self.MAX_CARACTERES_TEXTO:
            texto = texto[: self.MAX_CARACTERES_TEXTO]

        return (
            "Eres un analista de informacion delictual. Extrae datos "
            "estructurados de la siguiente noticia.\n\n"
            "REGLAS GENERALES\n"
            "- Extrae solo informacion explicita en el texto. No infieras "
            "ni completes con conocimiento externo.\n"
            "- No afirmes culpabilidad: usa el rol procesal que indique la "
            "noticia.\n"
            "- Si un dato no aparece, usa null o una lista vacia. Nunca "
            "rellenes con descripciones ni con texto inventado.\n"
            "- Devuelve exclusivamente JSON valido, sin markdown ni "
            "explicaciones.\n\n"
            f"CAMPOS OBLIGATORIOS: {campos}.\n\n"
            "REGLAS POR CAMPO\n\n"
            "fecha_publicacion: formato AAAA-MM-DD. Si la noticia no "
            "indica la fecha, usa null. No inventes una fecha ni uses "
            "expresiones como 'ayer' o 'hoy'.\n\n"
            "personas: lista de objetos con claves 'nombre' y 'rol'.\n"
            "  - Incluye SOLO a quienes intervienen en el hecho delictual: "
            "victima, imputado, detenido, acusado, condenado, testigo "
            "presencial o profugo.\n"
            "  - NO incluyas a quienes solo declaran o informan: fiscales, "
            "policias, seremis, delegados, alcaldes, jueces, abogados, "
            "peritos, voceros ni expertos. Sus instituciones van en "
            "'organizaciones'.\n"
            "  - 'nombre' debe ser un nombre propio tal como aparece en la "
            "noticia. Si la noticia no entrega el nombre, usa null en ese "
            "campo. Esta PROHIBIDO escribir descripciones como 'Hombre de "
            "29 anos', 'Sujeto detenido', 'Victima (hombre)' o 'imputado'.\n"
            f"  - 'rol' debe ser exactamente uno de: {self.ROLES_PERMITIDOS}.\n\n"
            f"delitos: lista de strings elegidos de: {self.DELITOS_PERMITIDOS}.\n"
            "  - Usa la categoria general y no la formula juridica literal: "
            "'homicidio calificado con alevosia' se registra como "
            "'homicidio'.\n"
            "  - Si el delito no encaja en ninguna categoria, usa 'otro'.\n\n"
            "organizaciones: lista de strings. Instituciones policiales, "
            "judiciales, de gobierno, bandas criminales, empresas u "
            "hospitales. Escribe cada institucion una sola vez y con el "
            "mismo nombre si vuelve a aparecer.\n\n"
            "lugares: lista de strings. SOLO ubicaciones geograficas: "
            "comunas, ciudades, regiones, poblaciones, calles o sectores. "
            "Las instituciones (hospitales, comisarias, tribunales) van en "
            "'organizaciones', no aqui.\n\n"
            "objetos: lista de objetos con claves 'tipo', 'nombre', "
            "'cantidad' y 'unidad'. Solo elementos concretos vinculados al "
            "hecho: armas, drogas, vehiculos, dinero o especies "
            "incautadas. 'cantidad' debe ser numerica o null.\n\n"
            "relaciones: lista de objetos con claves 'origen', 'tipo' y "
            "'destino'.\n"
            "  - 'origen' y 'destino' deben ser EXACTAMENTE uno de los "
            "strings que ya escribiste en delitos, personas.nombre, "
            "organizaciones, lugares u objetos.nombre de este mismo JSON. "
            "Copialo caracter por caracter. Si un extremo no figura en esas "
            "listas, omite esa relacion.\n"
            f"  - 'tipo' debe ser exactamente uno de: {self.RELACIONES_PERMITIDAS}.\n"
            "  - Cada relacion debe estar respaldada por una afirmacion "
            "explicita del texto.\n\n"
            "DATOS DE LA NOTICIA (usalos tal cual en el JSON)\n"
            f"id_noticia: {noticia.id_noticia}\n"
            f"fuente: {noticia.fuente}\n"
            f"url: {noticia.url}\n\n"
            "NOTICIA:\n"
            f"{texto}\n"
        )

    def extraer(self, noticia: NoticiaFuente) -> dict:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "Falta GEMINI_API_KEY. Copie .env.example a .env y complete la clave. "
                "Nunca suba .env a GitHub."
            )

        ruta = self.dir_json / f"{noticia.id_noticia}.json"
        if self.reusar_existentes:
            existente = self._leer_json_existente(ruta)
            if existente is not None:
                print("    Ya existe JSON; se reutiliza (no se llama a Gemini).")
                return existente

        bruto = self._llamar_con_reintentos(noticia)
        data = self._parsear_json(bruto)
        data["id_noticia"] = noticia.id_noticia
        if not data.get("fuente"):
            data["fuente"] = noticia.fuente
        if not data.get("url"):
            data["url"] = noticia.url
        ruta.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        time.sleep(PAUSA_ENTRE_REQUESTS)
        return data

    def _llamar_con_reintentos(self, noticia: NoticiaFuente) -> str:
        """Llama a Gemini reintentando ante errores transitorios.

        La API responde 503 cuando el modelo está saturado y 429 cuando se
        supera la cuota por minuto. Ambos son temporales: reintentar con una
        espera que se duplica (4s, 8s, 16s...) recupera la mayoría de las
        noticias que de otro modo se perderían. Un error no transitorio,
        como una clave inválida, se propaga de inmediato: reintentarlo sería
        inútil.
        """
        from google.genai import types

        cliente = self._obtener_cliente()
        configuracion = types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            response_mime_type="application/json",
            temperature=0,
        )

        espera = self.ESPERA_INICIAL
        ultimo_error: Exception | None = None

        for intento in range(1, self.INTENTOS_MAXIMOS + 1):
            try:
                respuesta = cliente.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=self.construir_prompt(noticia),
                    config=configuracion,
                )
                bruto = (getattr(respuesta, "text", None) or "").strip()
                if not bruto:
                    raise ValueError(
                        f"Gemini devolvió una respuesta vacía para {noticia.id_noticia}."
                    )
                return bruto
            except Exception as exc:  # noqa: BLE001 — se filtra por tipo de error abajo
                ultimo_error = exc
                if not self._es_transitorio(exc) or intento == self.INTENTOS_MAXIMOS:
                    raise
                print(
                    f"    Intento {intento}/{self.INTENTOS_MAXIMOS} falló "
                    f"({self._resumen_error(exc)}); reintentando en {espera:.0f}s..."
                )
                time.sleep(espera)
                espera *= 2

        raise ultimo_error if ultimo_error else RuntimeError("Fallo desconocido.")

    @classmethod
    def _es_transitorio(cls, exc: Exception) -> bool:
        """True si el error es de saturación o cuota, y conviene reintentar."""
        mensaje = str(exc).lower()
        return any(marca.lower() in mensaje for marca in cls.ERRORES_TRANSITORIOS)

    @staticmethod
    def _resumen_error(exc: Exception) -> str:
        mensaje = str(exc).replace("\n", " ")
        return mensaje[:90]

    @staticmethod
    def _leer_json_existente(ruta: Path) -> dict | None:
        """Devuelve el JSON ya guardado si existe y es válido; si no, None."""
        if not ruta.exists():
            return None
        try:
            data = json.loads(ruta.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else None

    def _obtener_cliente(self):
        if self._cliente is None:
            from google import genai

            self._cliente = genai.Client(api_key=GEMINI_API_KEY)
        return self._cliente

    @staticmethod
    def _parsear_json(bruto: str) -> dict:
        texto = bruto.strip()
        cerca = re.search(r"```(?:json)?\s*(.*?)\s*```", texto, re.DOTALL)
        if cerca:
            texto = cerca.group(1)
        try:
            data = json.loads(texto)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Gemini no devolvió JSON válido: {exc}. "
                f"Respuesta: {texto[:300]!r}"
            ) from exc
        if not isinstance(data, dict):
            raise ValueError("La respuesta de Gemini no es un objeto JSON.")
        return data
