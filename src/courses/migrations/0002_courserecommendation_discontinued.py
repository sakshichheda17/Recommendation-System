# Generated by Django 3.1.7 on 2021-02-28 11:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('courses', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='courserecommendation',
            name='discontinued',
            field=models.BooleanField(default=False),
        ),
    ]
