from django.db import models
from systemuser.models import SystemUser
# Create your models here.
class JobRecommendation(models.Model):
    user = models.ForeignKey(SystemUser, on_delete=models.CASCADE)
    job_index = models.SmallIntegerField(null=False)
    not_interested = models.BooleanField(default=False) 
    applied = models.BooleanField(default=False)
    rejected = models.BooleanField(default=False)
    selected = models.BooleanField(default=False)

    def __str__(self):
         return self.user.username + ' ' + str(self.job_index) + ' NI: ' + str(self.not_interested) + ' A: ' + str(self.applied)+' R: ' + str(self.rejected)+' S: ' + str(self.selected)