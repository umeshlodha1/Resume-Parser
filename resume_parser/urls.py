from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('resume/', include('resume_parser2.urls')),
    path('', include('resume_parser2.urls')),
]
