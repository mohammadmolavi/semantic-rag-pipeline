from django.contrib import admin, messages
from django.utils.html import format_html

from .models import Document, Question
from .services import generate_answer


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("title", "chunk_count", "created_at", "updated_at")
    search_fields = ("title", "text")
    readonly_fields = ("text", "chunk_count", "created_at", "updated_at")
    fields = ("title", "file", "text", "chunk_count", "created_at", "updated_at")


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ("short_question", "document", "created_at")
    list_filter = ("document", "created_at")
    search_fields = ("question", "answer")
    readonly_fields = ("answer", "formatted_sources", "created_at")
    fields = ("document", "question", "answer", "formatted_sources", "created_at")

    @admin.display(description="Question")
    def short_question(self, obj: Question) -> str:
        return str(obj)

    @admin.display(description="Sources")
    def formatted_sources(self, obj: Question) -> str:
        if not obj.sources:
            return "—"
        blocks = []
        for source in obj.sources:
            blocks.append(
                "chunk {chunk} | {origin}\n{content}".format(
                    chunk=source.get("chunk_index", "?"),
                    origin=source.get("source", "unknown"),
                    content=source.get("content", ""),
                )
            )
        return format_html(
            "<pre style=\"white-space: pre-wrap;\">{}</pre>",
            "\n\n---\n\n".join(blocks),
        )

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        try:
            generate_answer(obj)
            messages.success(request, "Answer generated from the indexed documents.")
        except Exception as error:
            messages.error(request, f"Could not generate an answer: {error}")
