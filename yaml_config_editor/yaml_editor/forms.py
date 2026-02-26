from django import forms


class LoginForm(forms.Form):
    username = forms.CharField(max_length=150)
    password = forms.CharField(widget=forms.PasswordInput)


class YamlForm(forms.Form):
    content = forms.CharField(
        widget=forms.Textarea(
            attrs={
                "rows": 30,
                "spellcheck": "false",
                "id": "yaml-textarea",
                "class": "form-control yaml-textarea",
            }
        )
    )
