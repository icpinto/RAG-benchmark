import hashlib, json

def stable_id(*parts: object, length: int = 16) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]
