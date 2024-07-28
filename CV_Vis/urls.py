from django.urls import path
from . import views


urlpatterns = [
    path('', views.input),
    path('process_canvas',views.process_canvas_data),
    path('animation', views.animation),
]