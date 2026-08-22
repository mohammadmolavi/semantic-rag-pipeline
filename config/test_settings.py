"""Lightweight settings used by the plain unittest regression suite."""

from .settings import *  # noqa: F403


DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}
INDEX_DOCUMENTS = False
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
