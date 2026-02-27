from django.urls import path
from .views import fetch_console_logs, yaml_editor_view

urlpatterns = [
    path("", yaml_editor_view, name="yaml_editor"),
    path("console-logs/", fetch_console_logs, name="fetch_console_logs"),
]
