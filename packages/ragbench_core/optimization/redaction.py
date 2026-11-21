"""Shared configuration redaction for prompts, artifacts and public APIs."""

import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


def public_configuration(value):
    """Copy saved settings without credentials or execution-only metadata."""
    secret_keys = {
        "apikey",
        "authorization",
        "password",
        "secret",
        "token",
        "accesstoken",
        "refreshtoken",
        "clientsecret",
        "accesskey",
        "secretaccesskey",
        "awsaccesskeyid",
        "awssecretaccesskey",
        "secretkey",
    }

    def secret(key):
        normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
        return normalized in secret_keys or normalized.endswith(
            ("apikey", "authorization", "password", "secret", "privatekey")
        )

    def redact(child):
        if isinstance(child, str) and re.fullmatch(
            r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", child
        ):
            return child
        return "[REDACTED]" if child else child

    if isinstance(value, dict):
        return {
            key: redact(child) if secret(key) else public_configuration(child)
            for key, child in value.items()
            if key not in {"_snapshot", "_optimization"}
        }
    if isinstance(value, list):
        return [public_configuration(child) for child in value]
    if isinstance(value, str) and "://" in value:
        try:
            parsed = urlparse(value)
            query = parse_qsl(parsed.query, keep_blank_values=True)
            if parsed.username is not None or any(secret(key) for key, _ in query):
                return urlunparse(
                    parsed._replace(
                        netloc=parsed.netloc.rsplit("@", 1)[-1],
                        query=urlencode(
                            [
                                (key, redact(child) if secret(key) else child)
                                for key, child in query
                            ]
                        ),
                    )
                )
        except ValueError:
            return "[REDACTED]"
    return value
