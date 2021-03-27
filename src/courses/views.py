from django.shortcuts import render
import pandas as pd
from .models import CourseRecommendation
from systemuser.models import SystemUser

# Create your views here.
def store_courses(courses,user):
    df = pd.read_csv('media/coursera_data.csv')
    
    for index,course in courses.items():
        course_rec = CourseRecommendation.objects.create(
            user = user,
            course_index = index
        )

def get_enrolled_courses(request):
    user_name = request.session['user_id']
    user = SystemUser.objects.get(username=user_name)

    if request.method == 'POST':
        form_data = request.POST
        print(form_data)
        if 'Discontinued' in form_data.values():
            course_index = list(form_data.keys())[list(form_data.values()).index('Discontinued')]
            change_course = CourseRecommendation.objects.filter(user=user).get(course_index=course_index)
            change_course.discontinued = True
            change_course.save()
        elif 'Completed' in form_data.values():
            course_index = list(form_data.keys())[list(form_data.values()).index('Completed')]
            change_course = CourseRecommendation.objects.filter(user=user).get(course_index=course_index)
            change_course.completed = True
            change_course.save()
		
    courses = CourseRecommendation.objects.filter(user=user)
    courses = courses.filter(enrolled=True).filter(completed=False).filter(discontinued=False)
    indices = [course.course_index for course in courses]
    
    df = pd.read_csv('media/coursera_data.csv')
    details = df.iloc[indices].to_dict(orient='list')
    
    column_names = list(details.keys())
    column_names = [x.replace('_',' ') for x in column_names]
    column_names.pop(0)
    column_names.insert(0,'Index')

    course_details = [list(a)[1:] for a in zip(*details.values())]
    for i in range(len(indices)):
        course_details[i].insert(0,indices[i])

    context = {'column_names':column_names, 'course_details':course_details}

    return render(request,'enrolled_courses.html',context)

def get_completed_courses(request):
    user_name = request.session['user_id']
    user = SystemUser.objects.get(username=user_name)
		
    courses = CourseRecommendation.objects.filter(user=user)
    courses = courses.filter(completed=True)
    indices = [course.course_index for course in courses]
    
    df = pd.read_csv('media/coursera_data.csv')
    details = df.iloc[indices].to_dict(orient='list')
    
    column_names = list(details.keys())
    column_names = [x.replace('_',' ') for x in column_names]
    column_names.pop(0)
    column_names.insert(0,'Index')

    course_details = [list(a)[1:] for a in zip(*details.values())]
    for i in range(len(indices)):
        course_details[i].insert(0,indices[i])

    context = {'column_names':column_names, 'course_details':course_details}

    return render(request,'completed_courses.html',context)