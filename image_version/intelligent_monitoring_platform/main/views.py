from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.conf import settings
from .models import MonitoringImage, Alarm, Record
from django.http import HttpResponse, HttpResponseServerError
import os
import base64
import time
import json


@login_required(login_url='login')
def index(request):
    alarms = Alarm.objects.order_by("-id").all()
    for alarm in alarms:
        alarm.alarm_class = "检测到人"
        alarm.time_begin = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(alarm.time_begin))
        alarm.time_end = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(alarm.time_end))
    return render(request, "index.html", {
        "alarms": alarms,
    })


@login_required(login_url='login')
def new_alarms(request):
    alarms = Alarm.objects.order_by("-id").all()
    result = []
    for alarm in alarms:
        if alarm.visible == 1:
            result.append({
                "id": alarm.id,
                "alarm_class": "检测到人",
                "time_begin": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(alarm.time_begin)),
                "time_end": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(alarm.time_end)),
            })
    return HttpResponse(json.dumps(result))


@login_required(login_url='login')
def delete_alarm(request, alarm_id):
    alarm_id = request.POST.get('alarm_id')
    Alarm.objects.filter(id=alarm_id).update(visible=0)
    return HttpResponse(json.dumps({
        "status": 200
    }))


@login_required(login_url='login')
def new_frame(request):
    try:
        all_objects = MonitoringImage.objects.order_by("id")
        image_info = all_objects[len(all_objects) - 1]
        image_path = os.path.join(settings.BASE_DIR, "%s.%s" % (
            image_info.cooked_image_filename, image_info.cooked_image_format))
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read())
        return HttpResponse(image_data, content_type="image/jpeg")
    except Exception as exception:
        print(exception)
        return HttpResponseServerError()


@login_required(login_url='login')
def current_status(request):
    try:
        all_objects = Record.objects.order_by("id")
        info = all_objects[len(all_objects) - 1].class_result
        return HttpResponse(info)
    except Exception as exception:
        print(exception)
        return HttpResponseServerError()
