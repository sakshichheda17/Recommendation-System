"""recommendation_system URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from systemuser.views import register,login,user_profile,jobs_rec,courses_rec
from jobs.views import get_applied_jobs,get_selected_jobs
from courses.views import get_enrolled_courses,get_completed_courses
from django.contrib.auth import views as auth_views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
	path('login/', login ,name='login'),
    path('register/', register,name='register'),      
    path('user_profile/',user_profile,name='user_profile'),
    path('jobs_rec/',jobs_rec,name='jobs_rec'),
    path('courses_rec/',courses_rec,name='courses_rec'),
    path('applied_jobs/',get_applied_jobs,name='applied_jobs'),
    path('selected_jobs/',get_selected_jobs,name='selected_jobs'),
    path('enrolled_courses/',get_enrolled_courses,name='enrolled_courses'),
    path('completed_courses/',get_completed_courses,name='completed_courses'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)