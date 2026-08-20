from rest_framework import serializers

from .models import Document, Question


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = (
            "id",
            "title",
            "file",
            "text",
            "chunk_count",
            "created_at",
            "updated_at",
        )
        read_only_fields = ("text", "chunk_count", "created_at", "updated_at")


class QuestionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Question
        fields = (
            "id",
            "document",
            "question",
            "answer",
            "sources",
            "created_at",
        )
        read_only_fields = ("answer", "sources", "created_at")


class AskSerializer(serializers.Serializer):
    question = serializers.CharField()
    document_id = serializers.IntegerField(required=False, allow_null=True)
    top_k = serializers.IntegerField(required=False, min_value=1, max_value=10, default=4)
