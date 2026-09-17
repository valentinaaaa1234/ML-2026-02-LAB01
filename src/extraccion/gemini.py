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

    def __init__(self, dir_json: Path = DIR_JSON, reusar_existentes: bool = True) -> None:
        self.dir_json = dir_json
        self.dir_json.mkdir(parents=True, exist_ok=True)
        self.reusar_existentes = reusar_existentes
        self._cliente = None

    def construir_prompt(self, noticia: NoticiaFuente) -> str:
        campos = ", ".join(self.CAMPOS_OBLIGATORIOS)
        texto = (noticia.texto_limpio or "").strip()
        if len(texto) > self.MAX_CARACTERES_TEXTO:
            texto = texto[: self.MAX_CARACTERES_TEXTO]
        return (
            "Analiza la siguiente noticia delictual.\n\n"
            "Extrae solamente informacion explicita. No inventes datos, "
            "entidades, roles ni relaciones.\n"
            "Devuelve exclusivamente JSON valido, sin markdown ni explicaciones.\n\n"
            f"Campos obligatorios: {campos}.\n"
            "personas: lista de objetos con claves nombre y rol.\n"
            "objetos: lista de objetos con claves tipo, nombre, cantidad, unidad.\n"
            "relaciones: lista de objetos con claves origen, tipo, destino.\n"
            "Si un dato no aparece, usa null o una lista vacia.\n\n"
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
