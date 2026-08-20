from django.contrib import admin, messages

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
    readonly_fields = ("answer", "sources", "created_at")
    fields = ("document", "question", "answer", "sources", "created_at")

    @admin.display(description="Question")
    def short_question(self, obj: Question) -> str:
        return str(obj)

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        try:
            generate_answer(obj)
            messages.success(request, "Answer generated from the indexed documents.")
        except Exception as error:
            messages.error(request, f"Could not generate an answer: {error}")
