from django.db import models
from django import forms
from django.core.validators import MinLengthValidator

class SystemUser(models.Model):
	username = models.CharField(max_length=80)
	password = models.CharField(max_length=15,validators=[MinLengthValidator(8)])
	email    = models.EmailField()
	user_resume = models.FileField(null=True,blank=True) 