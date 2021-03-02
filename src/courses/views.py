from django.shortcuts import render
import pandas as pd
from .models import CourseRecommendation

# Create your views here.
def store_courses(courses,user):
    df = pd.read_csv('media/coursera_data.csv')
    
    for index,course in courses.items():
        course_rec = CourseRecommendation.objects.create(
            user = user,
            course_index = index
        )