# Generated by Django 3.1.7 on 2021-02-28 08:59

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('systemuser', '0001_initial'),
        ('jobs', '0001_initial'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='JobRecommendations',
            new_name='JobRecommendation',
        ),
    ]
