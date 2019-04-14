from django.contrib import auth
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required, user_passes_test
from django.shortcuts import render, redirect


def login(request):
    if request.user.is_authenticated:
        return redirect("index")
    return render(request, "login.html")


def authenticate(request):
    username = request.POST.get("username")
    password = request.POST.get("password")
    user = auth.authenticate(request, username=username, password=password)
    if user is None:
        return redirect("login")
    auth.login(request, user)
    return redirect("index")


@login_required(login_url='login')
def logout(request):
    auth.logout(request)
    return redirect("login")


def user_is_admin(user):
    return user.is_superuser


@login_required(login_url='login')
@user_passes_test(user_is_admin)
def manage(request):
    users = User.objects.all()
    return render(request, "manage.html", {
        "users": users,
    })


@login_required(login_url='login')
def change_password(request):
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
    return redirect("index")


@login_required(login_url='login')
@user_passes_test(user_is_admin)
def add_user(request):
    username = request.POST.get("username")
    password = request.POST.get("password")
    user = User.objects.filter(username=username)
    if password != "":
        try:
            User.objects.create_user(username=username, password=password)
        except:
            pass
    return redirect("manage")
