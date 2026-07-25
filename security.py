import re

_URI_CREDENTIALS = re.compile(
    r"(?P<scheme>mongodb(?:\+srv)?://)[^@\s/]*@",
    re.IGNORECASE,
)

MAX_QUERY_LENGTH = 2000


def redact_credentials(value):
    """Replace inline credentials in MongoDB URIs with placeholders."""
    return _URI_CREDENTIALS.sub(r"\g<scheme>***:***@", str(value))


def sanitize_query(user_query, max_length=MAX_QUERY_LENGTH):
    """Validate and normalise a user supplied query string."""
    if not isinstance(user_query, str) or not user_query.strip():
        raise ValueError("User query cannot be empty")

    query = user_query.strip()
    if len(query) > max_length:
        raise ValueError(f"User query exceeds {max_length} characters")

    if "\x00" in query:
        raise ValueError("User query contains null bytes")

    return query
