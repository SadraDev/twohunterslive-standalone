from django.urls import path
from .views import yaml_editor_view

urlpatterns = [
    path("", yaml_editor_view, name="yaml_editor"),
]
