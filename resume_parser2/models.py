from django.db import models

class JobPosting(models.Model):
    job_description = models.TextField()
    min_experience = models.IntegerField()
    max_experience = models.IntegerField()
    comp_vertical = models.CharField(max_length=255)
    job_skills = models.JSONField(null=True, blank=True)
    comp_industry = models.CharField(max_length=255)
    job_location = models.CharField(max_length=255)

    def __str__(self):
        return self.job_description
