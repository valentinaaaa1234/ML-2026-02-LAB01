"""Limpieza de HTML a texto plano, según la guía del laboratorio."""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag


class LimpiadorHTML:
    """Quita ruido (menús, scripts, publicidad) y deja párrafos útiles."""

    ETIQUETAS_RUIDO = (
        "script",
        "style",
        "nav",
        "footer",
        "aside",
        "noscript",
        "iframe",
        "form",
        "button",
    )
    LARGO_MINIMO_LINEA = 40

    def limpiar(self, html: str) -> str:
        """HTML crudo → texto listo para enviar a un LLM."""
        soup = BeautifulSoup(html, "lxml")
        for etiqueta in soup(list(self.ETIQUETAS_RUIDO)):
            etiqueta.decompose()

        raiz = self._cuerpo_articulo(soup) or soup
        texto = raiz.get_text("\n")

        lineas = [linea.strip() for linea in texto.splitlines()]
        lineas = [linea for linea in lineas if len(linea) > self.LARGO_MINIMO_LINEA]
        return "\n".join(self._sin_duplicados(lineas))

    def _cuerpo_articulo(self, soup: BeautifulSoup) -> Tag | None:
        """Devuelve el <article> con más texto, si el sitio usa esa etiqueta.

        Muchos medios (WordPress en particular) encierran la noticia en un
        <article> y dejan las noticias relacionadas fuera de él. Quedarse
        con el artículo más extenso descarta esos titulares ajenos, que
        de otro modo el LLM podría interpretar como parte del hecho.
        """
        articulos = soup.find_all("article")
        if not articulos:
            return None
        return max(articulos, key=lambda art: len(art.get_text(" ", strip=True)))

    @staticmethod
    def _sin_duplicados(lineas: list[str]) -> list[str]:
        """Elimina líneas repetidas conservando el orden de aparición.

        El título suele aparecer varias veces (etiqueta <title>, encabezado
        y compartir en redes). Repetirlo no aporta información al LLM y
        sesga el resumen.
        """
        vistas: set[str] = set()
        unicas: list[str] = []
        for linea in lineas:
            if linea not in vistas:
                vistas.add(linea)
                unicas.append(linea)
        return unicas
