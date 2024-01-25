import os
import tempfile
import pdfkit
import requests
from django.shortcuts import render
from django.http import JsonResponse
from pyresparser import ResumeParser
from jinja2 import Template
from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import PdfUploadSerializer,JobPostingSerializer,ResumeOnlySerializer,ResumeJdSerializer,JdJsonSerializer
from .models import JobPosting
from rest_framework import status
from .analysis import analyze_job_match,final_analysis, calculate_overall_score, calculate_duration_in_months, get_industry_for_company, assign_score


class JobPostingView(APIView):
    def post(self, request, *args, **kwargs):
        # Clear all data from the database before saving
        self.clear_database()

        serializer = JobPostingSerializer(data=request.data)

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def clear_database(self):
        # Assuming JobPosting is the model you want to clear data for
        JobPosting.objects.all().delete()

class PdfUploadView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = PdfUploadSerializer(data=request.data)

        if serializer.is_valid():
            pdf_file = serializer.validated_data['pdf_file']
            
            try:
                # Save the uploaded file to a temporary location
                temp_file_path = os.path.join(tempfile.gettempdir(), pdf_file.name)

                with open(temp_file_path, 'wb') as temp_file:
                    for chunk in pdf_file.chunks():
                        temp_file.write(chunk)

                data = ResumeParser(temp_file_path).get_extracted_data()

                data['raw_data'] = temp_file_path

                # Fetch job details from the database
                job_posting = JobPosting.objects.last()  # You might want to fetch the relevant job posting based on your logic
                job_description = job_posting.job_description
                job_experience_range = (job_posting.min_experience, job_posting.max_experience)
                comp_vertical = job_posting.comp_vertical
                job_skills = job_posting.job_skills
                comp_industry = job_posting.comp_industry
                job_location = job_posting.job_location

                final_analysis_results = final_analysis(data, job_description,job_experience_range,comp_vertical,job_skills,comp_industry,job_location)
                data['final_analysis_results'] = final_analysis_results

                # Clean up: remove the temporary file after processing
                os.remove(temp_file_path)

                return Response({'result': data})
            except Exception as e:
                os.remove(temp_file_path)
                return JsonResponse({'success': False, 'error': str(e)})
        else:
            return Response(serializer.errors, status=400)


class ResumeJdView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ResumeJdSerializer(data=request.data)

        if serializer.is_valid():
            pdf_url = serializer.validated_data['pdf_file']
            print(pdf_url)
            try:
                # Save the uploaded file to a temporary location
                # pdf_content = "D:\Downloads\Arunkumaryadav.pdf"
                
                data = ResumeParser(pdf_url).get_extracted_data()

                data['raw_data'] = pdf_url

                # Fetch job details from the api
                job_posting_data = request.data.get('job_posting', {})
                job_description = job_posting_data.get('job_description', '')
                job_experience_range = (job_posting_data.get('min_experience', 0), job_posting_data.get('max_experience', 0))
                comp_vertical = job_posting_data.get('comp_vertical', '')
                job_skills = job_posting_data.get('job_skills', [])
                comp_industry = job_posting_data.get('comp_industry', '')
                job_location = job_posting_data.get('job_location', '')

                final_analysis_results = final_analysis(data, job_description,job_experience_range,comp_vertical,job_skills,comp_industry,job_location)
                data['final_analysis_results'] = final_analysis_results

                # Clean up: remove the temporary file after processing
               

                return Response({'result': data})
            except Exception as e:
               
                return JsonResponse({'success': False, 'error': str(e)})
        else:
            return Response(serializer.errors, status=400)
        


class JdJsonView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = JdJsonSerializer(data=request.data)

        if serializer.is_valid():
            
            try:
               
                data = request.data.get('pdf_file')
                # Fetch job details from the api
                job_posting_data = request.data.get('job_posting', {})
                job_description = job_posting_data.get('job_description', '')
                job_experience_range = (job_posting_data.get('min_experience', 0), job_posting_data.get('max_experience', 0))
                comp_vertical = job_posting_data.get('comp_vertical', '')
                job_skills = job_posting_data.get('job_skills', [])
                comp_industry = job_posting_data.get('comp_industry', '')
                job_location = job_posting_data.get('job_location', '')

                final_analysis_results = final_analysis(data, job_description,job_experience_range,comp_vertical,job_skills,comp_industry,job_location)
                data['final_analysis_results'] = final_analysis_results

                # Clean up: remove the temporary file after processing
                

                return Response({'result': final_analysis_results})
            except Exception as e:
               
                return JsonResponse({'success': False, 'error': str(e)})
        else:
            return Response(serializer.errors, status=400)

class ResumeOnlyView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ResumeOnlySerializer(data=request.data)

        if serializer.is_valid():
            pdf_file = serializer.validated_data['pdf_file']
            
            try:
                # Save the uploaded file to a temporary location
                temp_file_path = os.path.join(tempfile.gettempdir(), pdf_file.name)

                with open(temp_file_path, 'wb') as temp_file:
                    for chunk in pdf_file.chunks():
                        temp_file.write(chunk)

                data = ResumeParser(temp_file_path).get_extracted_data()

                data['raw_data'] = temp_file_path

                # Clean up: remove the temporary file after processing
                os.remove(temp_file_path)

                return Response({'resume': data})
            except Exception as e:
                os.remove(temp_file_path)
                return JsonResponse({'success': False, 'error': str(e)})
        else:
            return Response(serializer.errors, status=400)

def parse_resume(request):
    if request.method == 'POST' and request.FILES.get('resume'):
        resume_file = request.FILES['resume']

        # Create a temporary file in the system's temporary directory
        temp_file_path = os.path.join(tempfile.gettempdir(), resume_file.name)

        with open(temp_file_path, 'wb') as temp_file:
            for chunk in resume_file.chunks():
                temp_file.write(chunk)

        try:
            data = ResumeParser(temp_file_path).get_extracted_data()
            # create_resume_html(data=data)
            # create_resume_pdf()
            job_description = "Software Developer position in Mumbai requiring skills in Python, Django, and Linux, with at least 2 years of experience."
            job_experience_range = (2,5)
            comp_vertical = "IDBC"
            job_skills = ["Python", "Django", "Linux", "Programming", "C"]
            comp_industry = "IT"
            job_location = "Mumbai"

            final_analysis_results = final_analysis(data, job_description,job_experience_range,comp_vertical,job_skills,comp_industry,job_location)
            data['final_analysis_results'] = final_analysis_results

            os.remove(temp_file_path)  # Remove the temporary file after parsing
            return render(request, 'resume_analysis.html', {'data': data})
            # return JsonResponse({'success': True, 'data': data})
        except Exception as e:
            os.remove(temp_file_path)  # Remove the temporary file in case of an error
            return JsonResponse({'success': False, 'error': str(e)})

    
    return render(request, 'parse_resume.html')

def create_resume_html(data):
    # HTML template
    template_html = '''
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resume</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 20px;
        }

        .resume {
            background-color: #fff;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin: 20px;
        }

        h1, h2 {
            color: #333;
        }

        h2 {
            border-bottom: 2px solid #333;
            padding-bottom: 5px;
            margin-top: 10px;
        }

        p {
            line-height: 1.6;
        }

        .section {
            margin-bottom: 20px;
        }

        #skills {
            list-style-type: none;
            padding: 0;
        }

        #skills li {
            display: inline-block;
            margin: 5px;
            background-color: #007bff;
            color: #fff;
            padding: 5px 10px;
            border-radius: 3px;
        }
    </style>
      
</head>
<body>
    <div class="resume">
        <h1>{{ data.name }}</h1>
        <p>Email: {{ data.email }}</p>
        <p>Mobile Number: {{ data.mobile_number }}</p>
        <p>Gender: {{ data.gender }}</p>
        <p>Current City: {{ data.current_city }}</p>
        <p>{{ data.resume_objective }}</p>
        
        <p>Skills:</p>
        <ul id="skills">
            {% for skill in data.skills %}
                <li>{{ skill }}</li>
            {% endfor %}
        </ul>
        
        <div class="section">
            <h2>Education</h2>
            <p>Degree: {{ data.degree | default("Not Provided") }}</p>
            <p>College Name: {{ data.college_name | default("Not Provided") }}</p>
        </div>
        
        <div class="section">
            <h2>Experience</h2>
            <ul id="experience">
                {% for exp_key, exp_value in data.experience.items() %}
                    <li>{{ exp_value.designation }} at {{ exp_value.matched_text }}, {{ exp_value.duration }}</li>
                {% endfor %}
            </ul>
            <p>Company Names: {{ data.company_names | default("Not Provided") }}</p>
            <p>Total Experience: {{ data.total_experience | default("Not Provided") }} years</p>
        </div>

        
        <!-- Add more sections as needed -->
    </div>
</body>
</html>

    '''

    # Load the template
    template = Template(template_html)

    # Render HTML with JSON data
    html_content = template.render(data=data)

    # Save HTML to a file
    with open('resume.html', 'w', encoding='utf-8') as html_file:
        html_file.write(html_content)


def create_resume_pdf():
    # Configuration for wkhtmltopdf executable (provide the correct path)
    config = pdfkit.configuration(wkhtmltopdf='D:/mydocs/wkhtmltopdf/bin/wkhtmltopdf.exe')


    # Convert HTML to PDF
    pdfkit.from_file('resume.html', 'resume.pdf', configuration=config)

    print('Resume saved as resume.pdf')