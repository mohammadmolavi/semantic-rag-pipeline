#!/bin/sh
set -eu

echo "Applying database migrations..."
python manage.py migrate --noinput

echo "Creating or updating the admin user..."
python manage.py ensure_superuser

echo "Loading sample documents..."
python manage.py load_sample_data

echo "Starting Django development server..."
exec python manage.py runserver 0.0.0.0:8000