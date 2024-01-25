# serializers.py
from rest_framework import serializers
from .models import JobPosting

class JobPostingSerializer(serializers.ModelSerializer):
    class Meta:
        model = JobPosting
        fields = '__all__'

class PdfUploadSerializer(serializers.Serializer):
    pdf_file = serializers.FileField()


class ResumeOnlySerializer(serializers.Serializer):
    pdf_file = serializers.FileField()


class JdSerializer(serializers.Serializer):
    job_description = serializers.CharField()
    min_experience = serializers.IntegerField()
    max_experience = serializers.IntegerField()
    comp_vertical = serializers.CharField()
    job_skills = serializers.ListField(child=serializers.CharField())
    comp_industry = serializers.CharField()
    job_location = serializers.CharField()

class ResumeJdSerializer(serializers.Serializer):
    pdf_file = serializers.CharField()
    job_posting = JdSerializer(required=True)

class JdJsonSerializer(serializers.Serializer):
    pdf_file = serializers.JSONField()
    job_posting = JdSerializer(required=True)