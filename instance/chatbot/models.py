from django.db import models
import pickle

class Message(models.Model):
    user_uid = models.CharField(max_length=200)
    text = models.TextField()
    sender = models.CharField(max_length=50)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['timestamp']

class Document(models.Model):
    user_uid = models.CharField(max_length=200)
    filename = models.CharField(max_length=200)
    content = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

class Embedding(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    chunk = models.TextField()
    vector_data = models.BinaryField()