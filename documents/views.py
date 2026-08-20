from rest_framework import status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Document, Question
from .serializers import AskSerializer, DocumentSerializer, QuestionSerializer
from .services import generate_answer


class DocumentViewSet(viewsets.ModelViewSet):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer


class QuestionViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Question.objects.select_related("document")
    serializer_class = QuestionSerializer


class AskView(APIView):
    def post(self, request):
        serializer = AskSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        document = None
        document_id = data.get("document_id")
        if document_id:
            try:
                document = Document.objects.get(pk=document_id)
            except Document.DoesNotExist:
                return Response(
                    {"document_id": "Document not found."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        question = Question.objects.create(
            document=document,
            question=data["question"],
        )
        generate_answer(question, top_k=data.get("top_k", 4))
        return Response(
            QuestionSerializer(question).data,
            status=status.HTTP_201_CREATED,
        )
