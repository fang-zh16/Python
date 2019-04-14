from django.db import models


class MonitoringImage(models.Model):
    raw_image_filename = models.TextField(null=True)
    raw_image_format = models.TextField(null=True)
    cooked_image_filename = models.TextField(null=True)
    cooked_image_format = models.TextField(null=True)


class Record(models.Model):
    image = models.ForeignKey(MonitoringImage, on_delete=models.CASCADE)
    class_result = models.TextField(null=True)
    x = models.IntegerField(null=True)
    y = models.IntegerField(null=True)
    w = models.IntegerField(null=True)
    h = models.IntegerField(null=True)
    confidence = models.FloatField(null=True)
    time = models.BigIntegerField(null=True)


class Alarm(models.Model):
    alarm_class = models.TextField(null=True)
    image_id_begin = models.IntegerField(null=True)
    image_id_end = models.IntegerField(null=True)
    time_begin = models.BigIntegerField(null=True)
    time_end = models.BigIntegerField(null=True)
    visible = models.NullBooleanField()
