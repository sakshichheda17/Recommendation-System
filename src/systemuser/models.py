from django.db import models
from django import forms
from django.core.validators import MinLengthValidator

class SystemUser(models.Model):
	username = models.CharField(max_length=80)
	password = models.CharField(max_length=15,validators=[MinLengthValidator(8)])
	email    = models.EmailField(null=True,blank=True)
	user_resume = models.FileField(null=True,blank=True) 
	phone_number = models.CharField(max_length=20,null=True,blank=True)
	skills = models.CharField(max_length=900,null=True,blank=True)
	education = models.CharField(max_length=900,null=True,blank=True)

	# def delete(self,*args,**kwargs):
	# 	self.user_resume.delete()
	# 	super().delete(*args,**kwargs)

	def __str__(self):
         return self.username