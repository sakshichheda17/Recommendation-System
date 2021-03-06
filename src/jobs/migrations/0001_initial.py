# Generated by Django 3.1.7 on 2021-02-28 08:54

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('systemuser', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='JobRecommendations',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('job_index', models.SmallIntegerField()),
                ('not_interested', models.BooleanField(default=False)),
                ('applied', models.BooleanField(default=False)),
                ('selected', models.BooleanField(default=False)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='systemuser.systemuser')),
            ],
        ),
    ]
