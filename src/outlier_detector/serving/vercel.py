"""Restore the request path that the Vercel rewrite replaces.

``vercel.json`` rewrites every request to ``api/index.py`` and keeps the
original path in the ``__vercel_path`` query parameter. The middleware
below puts that path back before FastAPI routes the request.
"""

from typing import Any, Awaitable, Callable
from urllib.parse import parse_qsl, urlencode

ORIGINAL_PATH_PARAMETER = "__vercel_path"

AsgiApp = Callable[
    [
        dict[str, Any],
        Callable[[], Awaitable[Any]],
        Callable[[Any], Awaitable[None]],
    ],
    Awaitable[None],
]


def restore_original_path(scope: dict[str, Any]) -> dict[str, Any]:
    """Return a scope whose path comes from ``__vercel_path``.

    Parameters
    ----------
    scope : dict[str, Any]
        ASGI connection scope received from the Vercel runtime.

    Returns
    -------
    dict[str, Any]
        The same scope when the parameter is absent. Otherwise a copy with
        the original path, and without the parameter in the query string.

    Example
    -------
    >>> restore_original_path(
    ...     {"path": "/api/index.py", "query_string": b"__vercel_path=health"}
    ... )["path"]
    '/health'
    """
    query_pairs = parse_qsl(
        scope.get("query_string", b"").decode("latin-1"),
        keep_blank_values=True,
    )
    original_paths = [
        value for key, value in query_pairs if key == ORIGINAL_PATH_PARAMETER
    ]
    if not original_paths:
        return scope
    original_path = "/" + original_paths[0].lstrip("/")
    remaining_pairs = [
        (key, value)
        for key, value in query_pairs
        if key != ORIGINAL_PATH_PARAMETER
    ]
    return {
        **scope,
        "path": original_path,
        "raw_path": original_path.encode(),
        "query_string": urlencode(remaining_pairs).encode(),
    }


class RestoreOriginalPathMiddleware:
    """ASGI middleware that calls ``restore_original_path`` on HTTP requests.

    Parameters
    ----------
    app : AsgiApp
        Wrapped ASGI application.
    """

    def __init__(self, app: AsgiApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[Any]],
        send: Callable[[Any], Awaitable[None]],
    ) -> None:
        """Rewrite the path of HTTP and WebSocket scopes, then call the app.

        Parameters
        ----------
        scope : dict[str, Any]
            ASGI connection scope.
        receive : Callable[[], Awaitable[Any]]
            ASGI receive channel.
        send : Callable[[Any], Awaitable[None]]
            ASGI send channel.
        """
        if scope.get("type") in ("http", "websocket"):
            scope = restore_original_path(scope)
        await self.app(scope, receive, send)
