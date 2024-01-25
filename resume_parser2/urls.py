from django.urls import path
from .views import parse_resume
from .views import PdfUploadView,JobPostingView,ResumeOnlyView,ResumeJdView,JdJsonView
urlpatterns = [
    path('parse_resume/', parse_resume, name='parse_resume'),
    path('', parse_resume, name='parse_resume'),
    path('api/job_posting/', JobPostingView.as_view(), name='job_posting'),
    path('api/upload_pdf/', PdfUploadView.as_view(), name='upload_pdf'),
    path('api/resume_jd/', ResumeJdView.as_view(), name='resume_jd'),
    path('api/jd_json/', JdJsonView.as_view(), name='jd_json'),
    path('api/resume_only/', ResumeOnlyView.as_view(), name='resume_only')
]
