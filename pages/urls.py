# pages/urls.py
from django.urls import path, include
from .views import *

urlpatterns = [
    path('', form, name='form'),
    path('formProcess/', formProcess, name='formProcess'),
    path(
        'results/<str:size>/<str:weight>/<str:sweetness>/<str:softness>/<str:harvestTime>/<str:ripeness>/<str:acidity>/',
        results, name='results')
]
