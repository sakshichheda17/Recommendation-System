from django.shortcuts import render
import pandas as pd
from .models import JobRecommendation

# Create your views here.
def store_jobs(jobs,user):
    df = pd.read_csv('media/naukridataset.csv')
    df2 = df.head(50)
    
    for index,job in jobs.items():
        job_rec = JobRecommendation.objects.create(
            user = user,
            job_index = index
        )