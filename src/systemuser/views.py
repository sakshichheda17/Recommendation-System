from io import StringIO
from django.http import request
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate
from .forms import UserCreationForm, UserLoginForm,UserDetailsForm
from django.contrib import messages
from .models import SystemUser

from django.conf import settings 
from django.core.mail import send_mail 

from django.core.files.storage import FileSystemStorage

import docx2txt, nltk, re, requests
from nltk.corpus import stopwords

from pdfminer.high_level import extract_text

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# from resume_parser import resumeparse

# data = resumeparse.read_file('/path/to/resume/file')
# print(data)

from jobs.views import store_jobs
from courses.views import store_courses

from jobs.models import JobRecommendation
from courses.models import CourseRecommendation

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')

PHONE_REG = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
EMAIL_REG = re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')

SKILLS_DB = pd.read_csv('media/ResumeSkill.csv')
SKILLS_DB = SKILLS_DB["Skills"].tolist()

RESERVED_WORDS = [
    'school',
    'college',
    'university',
    'academy',
    'faculty',
    'institute',
	'education'
]

def login(request):
	if request.method == "POST":
		form = UserLoginForm(request.POST)        
		if form.is_valid():        	
			username = form.cleaned_data.get('username')        	
			password = form.cleaned_data.get('password')
			user = authenticate(username=username, password=password)            
			if user is not None:            	
				return redirect('/admin/')           
			else:            	
				if SystemUser.objects.filter(username=username,password=password).exists():            		
					messages.success(request, f'The user {username} was logged in successfully.')
					user_id = SystemUser.objects.get(username=username).id
					request.session['user_id'] = username     
					return redirect('user_profile')

				else:
					messages.warning(request,'Please enter a valid username and password.')

	else:
		form = UserLoginForm()


	return render(request,'login.html',{'form': form })


def register(request):
	if request.method == "POST":
		form = UserCreationForm(request.POST, request.FILES)

		if form.is_valid():
			username = form.cleaned_data.get('username')
			# email = form.cleaned_data.get('email')
			
			if SystemUser.objects.filter(username=form.cleaned_data['username']).exists():
				# print('Already exists')
				messages.warning(request, f'The username {username} already exists.')

			else:
				form.save()			
				#print('cleaned data',form.cleaned_data)
				messages.success(request, f'The user {username} was registered successfully.')
				request.session['user_id'] = username
				subject = 'Welcome to Recommendation System'
				message = f'Hi {username}, thank you for registering in Recommendation System.'
				# Uploading Resume
				uploaded_resume = request.FILES['user_resume']
				fs = FileSystemStorage()
				name = fs.save(uploaded_resume.name,uploaded_resume)
				# Extracting url of the uploaded pdf
				url = fs.url(name)
				resume_url = url.replace("%20"," ")
				resume_url = resume_url[1:]
				resume_url = resume_url.replace("\ ","/")

				# Extracting
				extraction(username,resume_url)

				return redirect('login')			    
			
	else:
	    form = UserCreationForm()	
			
	print(form.errors)
	return render(request,'register.html',{'form': form })



def extraction(username,resume_url):
	# Checking file extension and extracting text
	if resume_url.lower().endswith(('.pdf')):
		content = extract_text_from_pdf(resume_url)
	elif resume_url.lower().endswith(('.docx')) or resume_url.lower().endswith(('.doc')):
		content = extract_text_from_docx(resume_url)
	# Extracting names
	# names = extract_names(content)
	# print(names[0])
	# Extracting phone number
	phone_number = extract_phone_number(content)
	# print(phone_number)
	# Extracting email
	emails = extract_emails(content)
	# print(emails)
	# Extracting skills
	skills = extract_skills(content)
	# print(skills)
	
	# Extracting education
	education_information = extract_education(content)
	# print(education_information)

	# print("Recommending job for every skill")
	# jobs = set()
	jobs = dict()
	# Recommending job for every skill
	for skill in skills:
		jobs_dict = recommend_jobs(skill)
		# print(temp)
		# if len(temp) > 0:
		# 	for i in temp:
		# 		jobs.update(i)
		for index,job in jobs_dict.items():
			if len(job) > 0 and index not in jobs.keys():
				jobs[index] = job
	# jobs = list(jobs)
	
	print(jobs)
	user = SystemUser.objects.last()
	# store recommendations in job recommendations model
	store_jobs(jobs,user)

	print("Recommending course for every skill")
	courses = dict()
	# Recommending course for every skill
	for skill in skills:
		courses_dict = recommend_course(skill)
		# print(temp)
		# if len(temp)>0:
		# 	for i in temp:
		# 		courses.update(i)
		for index,course in courses_dict.items():
			if len(courses) == 10:
				break
			if len(course) > 0 and index not in courses.keys():
				courses[index] = course
				
	# courses = list(courses)
	print(courses)
	store_courses(courses,user)

	skill_list = list()
	for skill in skills:
		skill_list.append(skill)

	education_list = list()
	for edu in education_information:
		education_list.append(edu)

	user = SystemUser.objects.get(username=username)
	# print(user)
	user.email = emails[0]
	user.skills = skill_list
	user.phone_number = phone_number
	user.education = education_list
	user.save()

def user_profile(request):
	username = request.session['user_id']
	user = SystemUser.objects.get(username=username)
	context = {'user':user}
	# if user.skills:
	# 	skills = user.skills
	# 	context['skills'] = list(skills.split(","))

	if request.method == "POST": 
		form = UserDetailsForm(request.POST, request.FILES, instance = user) 
		context['form'] = form
		if form.is_valid():
			form.save()
			# Returns false if no new resume is uploaded
			new_resume = request.FILES.get('user_resume', False)
			
			# When new resume is uploaded
			if new_resume!= False:
			# new_reume = request.POST.get('user_resume')
				print("New resume",new_resume)
				fs = FileSystemStorage()
				url = fs.url(new_resume)
				# print("Before URL",url)
				# Deleting old Resume
				user.user_resume.delete()
				# Uploading Resume
				uploaded_resume = request.FILES['user_resume']
				# print("Uploading")
				name = fs.save(uploaded_resume.name,uploaded_resume)
				user.user_resume = name
				user.save()
				# print("Uploaded")
				# Extracting url of the uploaded pdf
				url = fs.url(name)
				# print("Uploaded URL",url)
				resume_url = url.replace("%20"," ")
				resume_url = resume_url[1:]
				resume_url = resume_url.replace("\ ","/")
				extraction(username,resume_url)

			
			messages.success(request, f'Your details were edited successfully.')
			return redirect("user_profile") 
		else:
			print(form.errors)

	return render(request,"user_profile.html",context)



def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def extract_text_from_docx(docx_path):
    txt = docx2txt.process(docx_path)
    if txt:
        return txt.replace('\t', ' ')
    return None

def extract_names(txt):
    person_names = []

    for sent in nltk.sent_tokenize(txt):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                person_names.append(
                    ' '.join(chunk_leave[0] for chunk_leave in chunk.leaves())
                )

    return person_names

def extract_phone_number(resume_text):
    phone = re.findall(PHONE_REG, resume_text)

    if phone:
        number = ''.join(phone[0])

        if resume_text.find(number) >= 0 and len(number) < 16:
            return number
    return None

def extract_emails(resume_text):
    return re.findall(EMAIL_REG, resume_text)

def extract_skills(input_text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    word_tokens = nltk.tokenize.word_tokenize(input_text)

    # remove the stop words
    filtered_tokens = [w for w in word_tokens if w not in stop_words]

    # remove the punctuation
    filtered_tokens = [w for w in word_tokens if w.isalpha()]

    # generate bigrams and trigrams (such as artificial intelligence)
    bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))

    # we create a set to keep the results in.
    found_skills = set()

    # we search for each token in our skills database
    for token in filtered_tokens:
        if token.lower() in SKILLS_DB:
            found_skills.add(token)

    # we search for each bigram and trigram in our skills database
    for ngram in bigrams_trigrams:
        if ngram.lower() in SKILLS_DB:
            found_skills.add(ngram)

    return found_skills


def extract_education(input_text):
    organizations = []

    # first get all the organization names using nltk
    for sent in nltk.sent_tokenize(input_text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if (hasattr(chunk, 'label') and chunk.label() == 'ORGANIZATION') or (hasattr(chunk, 'label') and chunk.label() == 'PERSON'):
                organizations.append(' '.join(c[0] for c in chunk.leaves()))

    # we search for each bigram and trigram for reserved words
    # (college, university etc...)
    education = set()
    for org in organizations:
        for word in RESERVED_WORDS:
            if org.lower().find(word) >= 0:
                education.add(org)

    return education


def recommend_jobs(skill):
	df = pd.read_csv('media/naukridataset.csv')
	df[['Job Title', 'Key Skills', 'Role Category', 'Location','Functional Area', 'Industry', 'Role']].groupby(['Location']).size().head(20)
	df2 = df.head(50)
	df2[df2.iloc[:, 5]=='NaN']
	pd.set_option("display.max_colwidth",None)
	tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
	tfidf_matrix = tf.fit_transform(df2['Key Skills'].apply(lambda x: np.str_(x)))
	len(df2) - df2['Job Title'].count()
	df2['Key Skills'].fillna("Unknown")
	cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
	skills = df2['Key Skills']
	titles = df2[['Job Title']]
	indices = pd.Series(df2.index, index=df2['Key Skills'])

	# Getting recommendations
	#     print([k for k,v in indices.items()])
	idx_list = [v for k,v in indices.items() if str(skill) in str(k)]
    
	job_indices = []
	
	for idx in idx_list:

		sim_scores = list(enumerate(cosine_sim[idx]))
		sim_scores = sorted(sim_scores,key=lambda x:x[1],reverse=True)

		job_indices += [i[0] for i in sim_scores if i[1]>0.9 and i[0] not in job_indices]
	# print(titles.iloc[job_indices][:10])
	jobs = titles.iloc[job_indices][:10]
	# print(jobs)

	jobs  = jobs.values.tolist()
	jobs_dict = {k:v for (k,v) in zip(job_indices, jobs)}
	
	return jobs_dict


def recommend_course(skill):
	df = pd.read_csv('media/coursera_data.csv')
	lst = [x for x in df['course_title'] if 'Python' in x]
	pd.set_option("display.max_colwidth",None)
	tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
	tfidf_matrix = tf.fit_transform(df['course_title'].apply(lambda x: np.str_(x)))
	len(df) - df['course_title'].count()
	cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
	titles = df[['course_title']]
	indices = pd.Series(df.index, index=df['course_title'])

	# Getting recommendations
	idx_list = [v for k,v in indices.items() if str(skill) in str(k)]
	course_indices = []
	for idx in idx_list:
		sim_scores = list(enumerate(cosine_sim[idx]))
		sim_scores = sorted(sim_scores,key=lambda x:x[1],reverse=True)
		course_indices += [i[0] for i in sim_scores if i[1]>0.9 and i[0] not in course_indices]
	
	# print(titles.iloc[course_indices][:10])
	courses = titles.iloc[course_indices][:10]
	# print(courses)
	courses = courses.values.tolist()
	courses_dict = {k:v for (k,v) in zip(course_indices, courses)}
	
	return courses_dict
	

def jobs_rec(request):
	user_name = request.session['user_id']
	user = SystemUser.objects.get(username=user_name)
	
	jobs = JobRecommendation.objects.filter(user=user)
	indices = [job.job_index for job in jobs]

	df = pd.read_csv('media/naukridataset.csv')
	df2 = df.head(50)

	#fetch job details using job index
	details = df2[['Job Title', 'Key Skills', 'Role Category', 'Location',
       'Functional Area', 'Industry', 'Role']].iloc[indices].to_dict(orient='list')
	
	job_titles = list(details.keys())
	
	job_details = list(zip(*details.values()))
	# print(job_details)
	
	context={'job_titles':job_titles, 'job_details':job_details}
	
	return render(request,'jobs_rec.html',context)

def courses_rec(request):
	user_name = request.session['user_id']
	user = SystemUser.objects.get(username=user_name)

	courses = CourseRecommendation.objects.filter(user=user)
	indices = [course.course_index for course in courses]
	df = pd.read_csv('media/coursera_data.csv')
	details = df.iloc[indices].to_dict(orient='list')
	course_titles = list(details.keys())
	course_titles = [x.replace('_',' ') for x in course_titles]
	course_details = list(zip(*details.values()))
	# print('\n'+str(len(course_details)))
	# for course in course_details:
		# print(course)
	context = {'course_titles':course_titles, 'course_details':course_details}

	return render(request,'courses_rec.html',context)