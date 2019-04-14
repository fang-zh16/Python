# Generated by Django 2.1.1 on 2018-09-11 17:16

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0002_auto_20180910_2031'),
    ]

    operations = [
        migrations.CreateModel(
            name='Record',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('class_result', models.TextField(null=True)),
                ('x', models.IntegerField(null=True)),
                ('y', models.IntegerField(null=True)),
                ('w', models.IntegerField(null=True)),
                ('h', models.IntegerField(null=True)),
                ('confidence', models.FloatField(null=True)),
                ('image', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='main.MonitoringImage')),
            ],
        ),
    ]
