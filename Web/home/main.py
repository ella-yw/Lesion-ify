from django.views.generic import TemplateView
from django.shortcuts import render

import sys; sys.path.insert(0, '../')
from Web_Predictor import pred

import keras, pandas, os

from django import forms
class uploadForm(forms.Form): file = forms.ImageField(widget = forms.ClearableFileInput(attrs={'class': 'inputfile inputfile-4', 'style': 'display:none'}))

from django.db import models
class imageModel(models.Model): model_pic = models.ImageField(upload_to = '')

class home(TemplateView):
    def get(self, request): return render(request, 'home/index.html', {'form': uploadForm()})
    def post(self, request):
        if request.method == 'POST':
            form = uploadForm(request.POST, request.FILES)
            if form.is_valid():
                imageModel(model_pic = form.cleaned_data['file']).save()
                msg_main, msg_all = pred(str(form.cleaned_data['file'].name), keras, pandas, os)
                return render(request, 'home/results.html', {'form': form, 'msg_main': msg_main, 'msg_all': msg_all})