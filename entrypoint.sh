#!/bin/sh
set -eu

docker compose up -d postgres

.venv/bin/python manage.py migrate --noinput
.venv/bin/python manage.py ensure_superuser
.venv/bin/python manage.py load_sample_data

exec .venv/bin/python manage.py runserver 0.0.0.0:8000
