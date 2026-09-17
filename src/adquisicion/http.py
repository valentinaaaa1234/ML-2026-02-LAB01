"""Cliente HTTP compartido: User-Agent, timeout y pausa entre peticiones."""

from __future__ import annotations

import time

import requests

from src.config import PAUSA_ENTRE_REQUESTS, TIMEOUT_HTTP, USER_AGENT


class ClienteHTTP:
    """Sesión HTTP educada para no sobrecargar servidores de prensa."""

    def __init__(
        self,
        timeout: int = TIMEOUT_HTTP,
        pausa: float = PAUSA_ENTRE_REQUESTS,
        user_agent: str = USER_AGENT,
    ) -> None:
        self.timeout = timeout
        self.pausa = pausa
        self.sesion = requests.Session()
        self.sesion.headers.update({"User-Agent": user_agent})

    def obtener(self, url: str, permitir_redirects: bool = True) -> requests.Response:
        """GET con pausa previa. Lanza HTTPError si el status no es 2xx."""
        time.sleep(self.pausa)
        respuesta = self.sesion.get(
            url,
            timeout=self.timeout,
            allow_redirects=permitir_redirects,
        )
        respuesta.raise_for_status()
        self._corregir_codificacion(respuesta)
        return respuesta

    @staticmethod
    def _corregir_codificacion(respuesta: requests.Response) -> None:
        """Evita acentos corruptos cuando el servidor no declara charset.

        Si la cabecera Content-Type no trae charset, requests asume
        ISO-8859-1 por compatibilidad con el estandar HTTP antiguo. La
        mayoria de los medios chilenos publica en UTF-8, por lo que esa
        suposicion produce texto como "aÃ±os" en vez de "años". En ese
        caso se deduce la codificacion real desde el contenido.
        """
        content_type = respuesta.headers.get("Content-Type", "").lower()
        if "charset=" not in content_type:
            respuesta.encoding = respuesta.apparent_encoding or "utf-8"

    def texto(self, url: str) -> str:
        """Cuerpo de la respuesta como texto."""
        return self.obtener(url).text

    def url_final(self, url: str) -> str:
        """Sigue redirecciones y devuelve la URL canónica del medio."""
        respuesta = self.obtener(url, permitir_redirects=True)
        return respuesta.url or url
