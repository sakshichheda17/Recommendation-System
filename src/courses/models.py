from django.db import models
from systemuser.models import SystemUser
# Create your models here.
class CourseRecommendation(models.Model):
    user = models.ForeignKey(SystemUser, on_delete=models.CASCADE)
    course_index = models.SmallIntegerField(null=False)
    not_interested = models.BooleanField(default=False) 
    enrolled = models.BooleanField(default=False)
    discontinued = models.BooleanField(default=False)
    completed = models.BooleanField(default=False)