from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView

admin.site.site_header = "Roshan Document Q&A"
admin.site.site_title = "Roshan Admin"
admin.site.index_title = "Document management"

urlpatterns = [
    path("", RedirectView.as_view(url="/admin/", permanent=False), name="home"),
    path("admin/", admin.site.urls),
    path("api/", include("documents.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
