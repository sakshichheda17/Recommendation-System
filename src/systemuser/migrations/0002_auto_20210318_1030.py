# Generated by Django 3.1.7 on 2021-03-18 05:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('systemuser', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='systemuser',
            name='education',
            field=models.CharField(blank=True, max_length=900, null=True),
        ),
        migrations.AddField(
            model_name='systemuser',
            name='phone_number',
            field=models.CharField(blank=True, max_length=20, null=True),
        ),
        migrations.AddField(
            model_name='systemuser',
            name='skills',
            field=models.CharField(blank=True, max_length=900, null=True),
        ),
        migrations.AlterField(
            model_name='systemuser',
            name='email',
            field=models.EmailField(blank=True, max_length=254, null=True),
        ),
    ]
