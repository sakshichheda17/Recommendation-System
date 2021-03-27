from django.shortcuts import render
import pandas as pd
from .models import JobRecommendation
from systemuser.models import SystemUser

# Create your views here.
def store_jobs(jobs,user):
    df = pd.read_csv('media/naukridataset.csv')
    df2 = df.head(50)
    
    for index,job in jobs.items():
        job_rec = JobRecommendation.objects.create(
            user = user,
            job_index = index
        )

def get_applied_jobs(request):
    user_name = request.session['user_id']
    user = SystemUser.objects.get(username=user_name)
    if request.method == 'POST':
        form_data = request.POST
        print(form_data)
        if 'Rejected' in form_data.values():
            job_index = list(form_data.keys())[list(form_data.values()).index('Rejected')]
            change_job = JobRecommendation.objects.filter(user=user).get(job_index=job_index)
            change_job.rejected = True
            change_job.save()
        elif 'Selected' in form_data.values():
            job_index = list(form_data.keys())[list(form_data.values()).index('Selected')]
            change_job = JobRecommendation.objects.filter(user=user).get(job_index=job_index)
            change_job.selected = True
            change_job.save()

    #  get all job rec for this user
    jobs = JobRecommendation.objects.filter(user=user)
    # filter jobs which the user has applied for and not yet selected
    jobs = jobs.filter(applied=True).filter(rejected=False).filter(selected=False)
    # get indices of these jobs
    indices = [job.job_index for job in jobs]

    df = pd.read_csv('media/naukridataset.csv')
    df2 = df.head(50)

    #fetch job details using job index
    details = df2[['Job Title', 'Key Skills', 'Role Category', 'Location',
        'Functional Area', 'Industry', 'Role']].iloc[indices].to_dict(orient='list')

    # get name of the columns for printing
    column_names = list(details.keys())
    column_names.insert(0,'Index')

    job_details = [list(a) for a in zip(*details.values())]
    #  insert index in job details
    for i in range(len(indices)):
        job_details[i].insert(0,indices[i])

    context={'indices': indices,'column_names':column_names, 'job_details':job_details}

    return render(request,'applied_jobs.html',context)

def get_selected_jobs(request):
    user_name = request.session['user_id']
    user = SystemUser.objects.get(username=user_name)

    #  get all job rec for this user
    jobs = JobRecommendation.objects.filter(user=user)
    # filter jobs which the user has applied for and not yet selected
    jobs = jobs.filter(selected=True)
    # get indices of these jobs
    indices = [job.job_index for job in jobs]

    df = pd.read_csv('media/naukridataset.csv')
    df2 = df.head(50)

    #fetch job details using job index
    details = df2[['Job Title', 'Key Skills', 'Role Category', 'Location',
        'Functional Area', 'Industry', 'Role']].iloc[indices].to_dict(orient='list')

    # get name of the columns for printing
    column_names = list(details.keys())
    column_names.insert(0,'Index')

    job_details = [list(a) for a in zip(*details.values())]
    #  insert index in job details
    for i in range(len(indices)):
        job_details[i].insert(0,indices[i])

    context={'indices': indices,'column_names':column_names, 'job_details':job_details}

    return render(request,'selected_jobs.html',context)