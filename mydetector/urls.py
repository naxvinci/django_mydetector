from django.urls import path
from .views import MyDetectorView

urlpatterns = [
    path('', MyDetectorView.as_view(), name="index"),
]
