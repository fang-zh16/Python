from django.contrib import auth
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required, user_passes_test
from django.shortcuts import render, redirect
from main.views import Static


def login(request):
    if request.user.is_authenticated:
        return redirect("localcam")
    return render(request, "login.html")


def authenticate(request):
    username = request.POST.get("username")
    password = request.POST.get("password")
    user = auth.authenticate(request, username=username, password=password)
    if user is None:
        return redirect("login")
    auth.login(request, user)
    return redirect("localcam")


@login_required(login_url='login')
def logout(request):
    while Static.thread_queue.qsize() != 0:
        Static.thread_queue.get()[1].put("quit signal")
    auth.logout(request)
    return redirect("login")


def user_is_admin(user):
    return user.is_superuser


@login_required(login_url='login')
@user_passes_test(user_is_admin)
def manage(request):
    while Static.thread_queue.qsize() != 0:
        Static.thread_queue.get()[1].put("quit signal")
    users = User.objects.all()
    return render(request, "manage.html", {
        "users": users,
    })


@login_required(login_url='login')
def change_password(request):
    while Static.thread_queue.qsize() != 0:
        Static.thread_queue.get()[1].put("quit signal")
    return render(request, "change_password.html", {
        "user_name": request.user,
    })


@login_required(login_url='login')
def change_password_done(request):
    if request.POST.get("new_password_1") != request.POST.get("new_password_2") or \
            request.POST.get("new_password_2") == "":
        return redirect("change_password")
    username = request.POST.get("username")
    password = request.POST.get("new_password_1")
    user = User.objects.get(username=username)
    user.set_password(password)
    user.save()
    return redirect("localcam")


@login_required(login_url='login')
@user_passes_test(user_is_admin)
def add_user(request):
    username = request.POST.get("username")
    password = request.POST.get("password")
    user = User.objects.filter(username=username)
    if len(user) == 0 and password != "":
        try:
            User.objects.create_user(username=username, password=password)
        except:
            pass
    return redirect("manage")
