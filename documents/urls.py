from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import AskView, DocumentViewSet, QuestionViewSet

router = DefaultRouter()
router.register("documents", DocumentViewSet, basename="document")
router.register("history", QuestionViewSet, basename="history")

urlpatterns = [
    path("ask/", AskView.as_view(), name="ask"),
    path("", include(router.urls)),
]
