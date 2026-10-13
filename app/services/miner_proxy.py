"""Adapt the miner's root-based dashboard to an authenticated per-instance URL."""

import re
import json
from http.cookies import SimpleCookie
from urllib.parse import urlsplit

from fastapi import HTTPException
from starlette.requests import HTTPConnection
from app.core.config import settings


HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "trailers", "transfer-encoding", "upgrade", "host",
}


class MinerUIProxy:
    def __init__(self, instance_id: str, port: int):
        self.base = f"/api/instances/{instance_id}/ui"
        self.origin = f"http://127.0.0.1:{port}"

    @staticmethod
    def public_origin(connection: HTTPConnection) -> str:
        if settings.UI_PUBLIC_ORIGIN:
            return settings.UI_PUBLIC_ORIGIN
        # TLS may terminate at the existing Apache/nginx proxy.
        scheme = connection.headers.get("x-forwarded-proto", connection.url.scheme)
        scheme = {"ws": "http", "wss": "https"}.get(scheme, scheme)
        if scheme not in {"http", "https"}:
            raise HTTPException(403, "Invalid proxy scheme")
        return f"{scheme}://{connection.headers.get('host', '')}"

    def request_headers(self, connection: HTTPConnection) -> dict[str, str]:
        origin = connection.headers.get("origin")
        if origin is not None and origin != self.public_origin(connection):
            raise HTTPException(403, "Invalid dashboard origin")

        headers = {
            k: v for k, v in connection.headers.items()
            if k.lower() not in HOP_BY_HOP | {"authorization", "cookie", "origin", "forwarded"}
            and not k.lower().startswith("x-forwarded-")
        }
        headers["host"] = urlsplit(self.origin).netloc
        headers["accept-encoding"] = "identity"
        # Override httpx's shared cookie jar even when this browser has no session.
        headers["cookie"] = ""
        if origin is not None:
            headers["origin"] = self.origin
        # Backend JWTs and other application cookies must stay at the backend.
        if session := connection.cookies.get("tdm_session"):
            cookie = SimpleCookie()
            cookie["tdm_session"] = session
            headers["cookie"] = cookie.output(header="").strip()
        return headers

    def rewrite_url(self, url: str) -> str:
        if url.startswith(self.origin + "/"):
            url = url[len(self.origin):]
        if url.startswith("/") and not url.startswith("//"):
            if url != self.base and not url.startswith(self.base + "/"):
                return self.base + url
        return url

    def response_cookie(self, value: str, connection: HTTPConnection) -> list[str]:
        cookies = SimpleCookie()
        cookies.load(value)
        result = []
        for cookie in cookies.values():
            cookie["path"] = self.rewrite_url(cookie["path"] or "/")
            cookie["domain"] = ""
            if self.public_origin(connection).startswith("https://"):
                cookie["secure"] = True
            result.append(cookie.OutputString())
        return result

    def rewrite_javascript(self, source: str) -> str:
        # import() cannot be intercepted by the fetch shim. noVNC is loaded this
        # way in v2.2; its own relative module imports then resolve under /ui/.
        source = re.sub(
            r"(\bimport\s*\(\s*['\"])(/(?!/)[^'\"]*)(['\"]\s*\))",
            lambda m: m[1] + self.rewrite_url(m[2]) + m[3], source,
        )
        # Location methods are browser built-ins that cannot be monkey-patched.
        # Keep the dashboard's login/logout redirects inside the instance URL.
        return re.sub(
            r"((?:window\.)?location\.(?:replace|assign)\()([^;\n]*)(\);)",
            lambda m: m[1] + "window.__tdmProxyURL(" + m[2] + ")" + m[3], source,
        )

    def rewrite_html(self, html: str) -> str:
        html = re.sub(
            r"((?:href|src|action)\s*=\s*)(['\"])(/(?!/)[^'\"]*)\2",
            lambda m: m[1] + m[2] + self.rewrite_url(m[3]) + m[2], html,
        )
        html = self.rewrite_javascript(html)
        html = re.sub(
            r"(url\(\s*['\"]?)(/(?!/)[^)'\"\s]*)(['\"]?\s*\))",
            lambda m: m[1] + self.rewrite_url(m[2]) + m[3], html,
        )
        shim = """<script>
(function(){
  var base=__BASE__;
  function prefixed(p){return p===base||p.startsWith(base+'/');}
  function rw(value){
    var u=value instanceof URL?value.href:value;
    if(typeof u!=='string')return value;
    if(u.startsWith('/')&&!u.startsWith('//'))return prefixed(u)?u:base+u;
    try{
      var parsed=new URL(u,location.href);
      if(parsed.host===location.host&&
         (parsed.origin===location.origin||parsed.protocol==='ws:'||parsed.protocol==='wss:')&&
         !prefixed(parsed.pathname)){
        parsed.pathname=base+parsed.pathname;return parsed.href;
      }
    }catch(e){}
    return value;
  }
  window.__tdmProxyURL=rw;
  var fetchOriginal=window.fetch;
  window.fetch=function(input,options){
    if(input instanceof Request)input=new Request(rw(input.url),input);
    else input=rw(input);
    return fetchOriginal.call(window,input,options);
  };
  var openOriginal=XMLHttpRequest.prototype.open;
  XMLHttpRequest.prototype.open=function(method,url){
    arguments[1]=rw(url);return openOriginal.apply(this,arguments);
  };
  var WS=window.WebSocket;
  function PatchedWS(url,protocols){
    return protocols!==undefined?new WS(rw(url),protocols):new WS(rw(url));
  }
  PatchedWS.prototype=WS.prototype;
  ['CONNECTING','OPEN','CLOSING','CLOSED'].forEach(function(key){PatchedWS[key]=WS[key];});
  window.WebSocket=PatchedWS;
})();
</script>""".replace("__BASE__", json.dumps(self.base))
        # Run before auth.js and Socket.IO capture fetch/WebSocket references.
        if re.search(r"<head\b", html, re.IGNORECASE):
            return re.sub(r"<head\b[^>]*>", lambda m: m[0] + shim, html, count=1, flags=re.IGNORECASE)
        return shim + html
