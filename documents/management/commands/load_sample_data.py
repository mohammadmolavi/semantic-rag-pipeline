from django.conf import settings
from django.core.files import File
from django.core.management.base import BaseCommand

from documents.models import Document
from documents.sample_data import SAMPLE_DOCUMENTS, ensure_sample_files


class Command(BaseCommand):
    help = "Create and load the bundled multilingual demonstration documents."

    def handle(self, *args, **options):
        sample_dir = settings.SAMPLE_DATA_DIR
        ensure_sample_files(sample_dir)
        loaded = 0
        skipped = 0

        for specification in SAMPLE_DOCUMENTS:
            if Document.objects.filter(title=specification.title).exists():
                skipped += 1
                self.stdout.write(f"Already loaded: {specification.title}")
                continue

            path = sample_dir / specification.filename

            with path.open("rb") as handle:
                document = Document(title=specification.title)
                document.file.save(path.name, File(handle), save=True)

            loaded += 1
            self.stdout.write(
                self.style.SUCCESS(
                    f"Loaded sample document id={document.pk}: {specification.title}"
                )
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"Sample data ready: {loaded} loaded, {skipped} already present."
            )
        )
