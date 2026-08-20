from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models, transaction

from rag.loaders import SUPPORTED_SUFFIXES, load_document


def validate_document_file(upload) -> None:
    name = getattr(upload, "name", "")
    suffix = f".{name.rsplit('.', 1)[-1].lower()}" if "." in name else ""
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValidationError("Only .docx and .txt files are supported.")


class Document(models.Model):
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to="documents/", validators=[validate_document_file])
    text = models.TextField(blank=True)
    chunk_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.title

    @property
    def vector_source(self) -> str:
        return f"document:{self.pk}"

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        if self.file:
            extracted = load_document(self.file.path)
            if extracted != self.text:
                type(self).objects.filter(pk=self.pk).update(text=extracted)
                self.text = extracted
        transaction.on_commit(lambda pk=self.pk: _index_document(pk))

    def delete(self, *args, **kwargs):
        source = self.vector_source
        chunk_count = self.chunk_count
        file_name = self.file.name if self.file else ""
        storage = self.file.storage if self.file else None
        super().delete(*args, **kwargs)
        if file_name and storage:
            storage.delete(file_name)
        if settings.INDEX_DOCUMENTS:
            from rag.services import delete_source

            delete_source(source, chunk_count)


def _index_document(document_id: int) -> None:
    if not settings.INDEX_DOCUMENTS:
        return

    from rag.services import reindex_source

    document = Document.objects.filter(pk=document_id).first()
    if document is None:
        return
    chunk_count = reindex_source(
        document.vector_source,
        document.text,
        previous_chunk_count=document.chunk_count,
    )
    Document.objects.filter(pk=document_id).update(chunk_count=chunk_count)


class Question(models.Model):
    document = models.ForeignKey(
        Document,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="questions",
        help_text="Leave empty to search across all documents.",
    )
    question = models.TextField()
    answer = models.TextField(blank=True)
    sources = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        preview = self.question.strip().replace("\n", " ")
        return preview[:80] or f"Question {self.pk}"
