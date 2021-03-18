from django import forms
from .models import SystemUser


class UserCreationForm(forms.ModelForm):
    class Meta:
        model = SystemUser
        fields = [
        			'username',
        			'password',
        			'user_resume'
        	]
        widgets = {
        'password': forms.PasswordInput(),
    	}


class UserLoginForm(forms.ModelForm):
    class Meta:
        model = SystemUser
        fields = [
                    'username',
                    'password',
                ]
        widgets = {
            'password': forms.PasswordInput(),
            }


class UserDetailsForm(forms.ModelForm):
    class Meta:
        model = SystemUser
        fields = [
                    'username',
                    'email',
                    'phone_number',
                    'education',
                    'skills'
                ]