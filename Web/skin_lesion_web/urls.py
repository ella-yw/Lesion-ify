from django.contrib import admin
from django.urls import path
from home.main import home

urlpatterns = [
        path('', home.as_view()),
        path('admin/', admin.site.urls),
    ]
