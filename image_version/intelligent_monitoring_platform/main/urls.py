from django.urls import path
from . import views, authentication_views

urlpatterns = [
    path('', views.index, name="index"),
    path('new_frame', views.new_frame, name="new_frame"),
    path('new_alarms', views.new_alarms, name="new_alarms"),
    path('current_status', views.current_status, name="current_status"),
    path('delete_alarm/<int:alarm_id>', views.delete_alarm, name="delete_alarm"), 

    path('login', authentication_views.login, name="login"),
    path('authenticate', authentication_views.authenticate, name="authenticate"),
    path('logout', authentication_views.logout, name="logout"),
    path('manage', authentication_views.manage, name="manage"),
    path('change_password', authentication_views.change_password,
         name="change_password"),
    path('change_password_done', authentication_views.change_password_done,
         name="change_password_done"),
    path('add_user', authentication_views.add_user, name="add_user"),
]
