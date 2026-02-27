from pathlib import Path
import yaml
import json

from django.http import JsonResponse
from django.conf import settings
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_protect

from .forms import LoginForm, YamlForm


SESSION_KEY = "yaml_editor_authenticated"


def _get_yaml_path() -> Path:
    return Path(settings.YAML_FILE_PATH)

def _get_console_path() -> Path:
    return Path(settings.BASE_DIR).parent / "reports" / "logs"

def _read_yaml_text() -> str:
    path = _get_yaml_path()
    if not path.exists():
        return "# YAML file not found at: %s" % path
    return path.read_text(encoding="utf-8")

def fetch_console_logs(request):
    content, path = _get_console_logs()
    return JsonResponse({
        "log_content": content,
        "log_path": path,
    })

def _get_editor_credentials() -> tuple[str | None, str | None]:
    """
    Read web editor credentials from TwoHuntersLive_StandAlone.yaml.

    Expected structure:

        admin:
          username: "your_username"
          password: "your_password"
    """
    path = _get_yaml_path()
    if not path.exists():
        return None, None

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        web_cfg = data.get("admin", {}) or {}
        username = web_cfg.get("username")
        password = web_cfg.get("password")
        return username, password
    except Exception:
        return None, None


def _validate_yaml(text: str) -> tuple[bool, str | None]:
    try:
        yaml.safe_load(text)
        return True, None
    except yaml.YAMLError as exc:
        return False, str(exc)


def _flatten_dict(d, parent_key: str = "", sep: str = "."):
    """Flatten a nested dictionary for the friendly UI."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _unflatten_dict(d, sep: str = "."):
    """Unflatten a dictionary back to nested structure."""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        d_ref = result
        for part in parts[:-1]:
            if part not in d_ref:
                d_ref[part] = {}
            d_ref = d_ref[part]
        d_ref[parts[-1]] = value
    return result

def _get_console_logs() -> tuple[str | None, str]:
    """
    Find the newest log file in BASE_DIR/reports/logs/*.log and return its content.
    """
    log_dir = _get_console_path()
    if not log_dir.exists():
        return "# No logs directory found", str(log_dir)

    # Find newest .log file (YYYY-MM-DD.log pattern)
    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        return "# No .log files found", str(log_dir)

    newest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    try:
        content = newest_log.read_text(encoding="utf-8", errors="replace")
        return content[-10000:], str(newest_log)  # Last 10k chars for performance
    except Exception:
        return f"# Failed to read {newest_log}", str(newest_log)


@csrf_protect
def yaml_editor_view(request):
    authenticated = request.session.get(SESSION_KEY, False)

    message = None
    message_type = "info"

    # --- Logout ---
    if authenticated and request.method == "POST" and "logout" in request.POST:
        request.session.flush()
        return redirect("yaml_editor")

    # --- Login flow ---
    if not authenticated:
        if request.method == "POST":
            form = LoginForm(request.POST)
            if form.is_valid():
                username = form.cleaned_data["username"]
                password = form.cleaned_data["password"]

                valid_user, valid_pass = _get_editor_credentials()

                if valid_user is None or valid_pass is None:
                    message = (
                        "Editor credentials are not configured in the YAML file. "
                        "Please add an 'admin.username' and 'admin.password' block."
                    )
                    message_type = "danger"
                elif username == valid_user and password == valid_pass:
                    request.session[SESSION_KEY] = True
                    return redirect("yaml_editor")
                else:
                    message = "Invalid credentials."
                    message_type = "danger"
        else:
            form = LoginForm()

        return render(
            request,
            "editor.html",
            {
                "authenticated": False,
                "login_form": form,
                "yaml_form": None,
                "message": message,
                "message_type": message_type,
            },
        )

    # --- Authenticated: YAML editor ---
    if request.method == "POST":
        save_mode = request.POST.get("save_mode", "classic")

        if save_mode == "friendly":
            # Reconstruct YAML from flat form fields
            try:
                original_yaml_text = _read_yaml_text()
                yaml_data = yaml.safe_load(original_yaml_text) or {}

                flat_original = _flatten_dict(yaml_data)

                # Include new "field_*" keys that did not exist originally
                for form_key, form_val in request.POST.items():
                    if not form_key.startswith("field_"):
                        continue
                    key_path = form_key[len("field_") :]
                    if key_path not in flat_original:
                        flat_original[key_path] = form_val

                for key in list(flat_original.keys()):
                    orig_val = flat_original[key]

                    # Special handling: pause_symbol / resume_symbol support null via FLUSH
                    if key in (
                        "admin_commands.pause_symbol",
                        "admin_commands.resume_symbol",
                    ):
                        form_val = request.POST.get(f"field_{key}")
                        if form_val is not None:
                            if form_val == "__NULL__":
                                flat_original[key] = None
                            else:
                                flat_original[key] = form_val
                        continue

                    if isinstance(orig_val, bool):
                        if f"field_{key}" in request.POST:
                            val = request.POST.get(f"field_{key}")
                            flat_original[key] = val in ("true", "on", "1", "True")
                    elif isinstance(orig_val, int):
                        form_val = request.POST.get(f"field_{key}")
                        if form_val is not None:
                            flat_original[key] = int(form_val)
                    elif isinstance(orig_val, float):
                        form_val = request.POST.get(f"field_{key}")
                        if form_val is not None:
                            flat_original[key] = float(form_val)
                    elif isinstance(orig_val, str):
                        form_val = request.POST.get(f"field_{key}")
                        if form_val is not None:
                            flat_original[key] = form_val
                    else:
                        # None / unknown types can still be overridden as string
                        form_val = request.POST.get(f"field_{key}")
                        if form_val is not None:
                            flat_original[key] = form_val

                updated_data = _unflatten_dict(flat_original)
                text = yaml.safe_dump(
                    updated_data,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
                ok, err = _validate_yaml(text)
            except Exception as e:
                ok, err, text = False, str(e), ""
        else:
            # Classic: save raw text exactly, only validate
            yaml_form = YamlForm(request.POST)
            if yaml_form.is_valid():
                text = yaml_form.cleaned_data["content"]
                ok, err = _validate_yaml(text)
            else:
                ok, err = False, "Form invalid"

        if not ok:
            message = "YAML is invalid: %s" % err
            message_type = "danger"
            yaml_form = YamlForm(
                initial={"content": request.POST.get("content", _read_yaml_text())}
            )
        else:
            path = _get_yaml_path()
            try:
                path.write_text(text, encoding="utf-8")
                request.session["yaml_saved_ok"] = True
                yaml_form = YamlForm(initial={"content": text})
            except Exception as exc:
                message = "Failed to save file: %s" % exc
                message_type = "danger"
                yaml_form = YamlForm(initial={"content": text})
    else:
        initial_text = _read_yaml_text()
        yaml_form = YamlForm(initial={"content": initial_text})

    saved_ok = bool(request.session.pop("yaml_saved_ok", False))

    # Prepare Friendly Mode fields + status
    is_running = False
    friendly_fields = []

    log_content, log_path = _get_console_logs()

    try:
        current_text = _read_yaml_text()
        data = yaml.safe_load(current_text) or {}
        is_running = bool(data.get("system", {}).get("running", False))

        flat_data = _flatten_dict(data)

        # Admin one-shot actions (buttons)
        admin_keys = {
            "admin_commands.stop",
            "admin_commands.reset",
            "admin_commands.update",
            "admin_commands.say_hello",
            "admin_commands.pause",
            "admin_commands.resume",
        }

        for key, value in flat_data.items():
            # Skip system.running
            if key == "system.running":
                continue

            # Always treat pause_symbol / resume_symbol as editable strings
            if key in ("admin_commands.pause_symbol", "admin_commands.resume_symbol"):
                val = "" if value is None else str(value)
                friendly_fields.append(
                    {"key": key, "value": val, "type": "string"}
                )
                continue

            if key in admin_keys and isinstance(value, bool):
                friendly_fields.append(
                    {"key": key, "value": value, "type": "admin"}
                )
                continue

            if isinstance(value, bool):
                friendly_fields.append({"key": key, "value": value, "type": "bool"})
            elif isinstance(value, (int, float)):
                friendly_fields.append(
                    {"key": key, "value": value, "type": "number"}
                )
            elif isinstance(value, str):
                friendly_fields.append(
                    {"key": key, "value": value, "type": "string"}
                )
    except Exception:
        pass

    return render(
        request,
        "editor.html",
        {
            "authenticated": True,
            "login_form": None,
            "yaml_form": yaml_form,
            "message": message,
            "message_type": message_type,
            "yaml_path": str(_get_yaml_path()),
            "server_running": is_running,
            "friendly_fields": friendly_fields,
            "json_fields": json.dumps(friendly_fields),
            "log_content": log_content,
            "log_path": log_path,
            "saved_ok": saved_ok,
        },
    )
