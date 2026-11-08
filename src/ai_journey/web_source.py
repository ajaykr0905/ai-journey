"""Validated source identities for replayable public-web ingestion."""

from dataclasses import dataclass
import ipaddress
from urllib.parse import SplitResult, urlsplit, urlunsplit

from .bpe import TokenizerError


def canonical_public_url(value: str) -> str:
    """Return one fragment-free identity for a public HTTP(S) page."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise TokenizerError("web source URL must be a nonempty trimmed string")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise TokenizerError("web source URL is malformed") from exc
    if parsed.scheme not in {"http", "https"}:
        raise TokenizerError("web source URL must use HTTP or HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise TokenizerError("web source URL must not contain user information")
    hostname = parsed.hostname
    if hostname is None or not hostname:
        raise TokenizerError("web source URL requires a hostname")
    try:
        ascii_hostname = hostname.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise TokenizerError("web source hostname is invalid") from exc
    if ascii_hostname == "localhost" or ascii_hostname.endswith(".localhost"):
        raise TokenizerError("web source hostname must be public")
    try:
        address = ipaddress.ip_address(ascii_hostname)
    except ValueError:
        if "." not in ascii_hostname:
            raise TokenizerError("web source hostname must be fully qualified")
    else:
        if not address.is_global:
            raise TokenizerError("web source IP address must be globally routable")
        ascii_hostname = (
            f"[{ascii_hostname}]" if address.version == 6 else ascii_hostname
        )
    if port is not None and port != {"http": 80, "https": 443}[parsed.scheme]:
        raise TokenizerError("web source URL must use the scheme's default port")
    path = parsed.path or "/"
    return urlunsplit(
        SplitResult(parsed.scheme, ascii_hostname, path, parsed.query, "")
    )


@dataclass(frozen=True)
class WebPage:
    """Immutable downloaded page bytes paired with canonical provenance."""

    url: str
    content_type: str
    payload: bytes

    def __post_init__(self) -> None:
        canonical = canonical_public_url(self.url)
        if canonical != self.url:
            raise TokenizerError("web page URL must already be canonical")
        if not isinstance(self.content_type, str) or not self.content_type.strip():
            raise TokenizerError("web page content type must be nonempty")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise TokenizerError("web page payload must be nonempty bytes")
