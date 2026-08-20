#!/bin/sh
set -eu

python manage.py migrate --noinput
python manage.py ensure_superuser
python manage.py load_sample_data

exec python manage.py runserver 0.0.0.0:8000
