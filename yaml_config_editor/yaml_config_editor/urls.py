from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("yaml_editor.urls")),  # root goes to our single-page editor
]
