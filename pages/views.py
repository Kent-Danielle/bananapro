import pickle
import dill
import pandas as pd
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponseRedirect
from django.http import Http404
from django.urls import reverse
from django.shortcuts import render, redirect


def form(request):
    return render(request, 'form.html', {
        'mynumbers': [1, 2, 3, 4, 5, 6, ],
        'firstName': 'Kent',
        'lastName': 'Concengco'})


def formProcess(request):
    try:
        size = request.POST['size']
        weight = request.POST['weight']
        sweetness = request.POST['sweetness']
        softness = request.POST['softness']
        harvest_time = request.POST['harvestTime']
        ripeness = request.POST['ripeness']
        acidity = request.POST['acidity']
    except:
        return render(request, 'form.html', {
            'errorMessage': 'Invalid input. Please try again.'})
    else:
        return HttpResponseRedirect(
            reverse('results', kwargs={
                'size': size,
                'weight': weight,
                'sweetness': sweetness,
                'softness': softness,
                'harvestTime': harvest_time,
                'ripeness': ripeness,
                'acidity': acidity
            }
                    )
        )


def results(request, size, weight, sweetness, softness, harvestTime, ripeness, acidity):
    with open('C:\\Users\\kentc\\Documents\\Term 4\\BigData_COMP4949\\Sandbox\\hw2\\knn_pkl', 'rb') as f:
        model = pickle.load(f)

    with open('C:\\Users\\kentc\\Documents\\Term 4\\BigData_COMP4949\\Sandbox\\hw2\\scaler_pkl', 'rb') as f:
        scaler = pickle.load(f)

    values = [[size, weight, sweetness, softness, harvestTime, ripeness, acidity]]
    values = scaler.transform(values)
    single_sample_df = pd.DataFrame(
        columns=['Size', 'Weight', 'Sweetness', 'Softness', 'HarvestTime', 'Ripeness', 'Acidity'],
        data=values)
    print(single_sample_df.head())
    prediction = model.predict(single_sample_df)
    print(prediction)
    return render(request, 'results.html', {
        'size': size,
        'weight': weight,
        'sweetness': sweetness,
        'softness': softness,
        'harvestTime': harvestTime,
        'ripeness': ripeness,
        'acidity': acidity,
        'prediction': prediction[0]
    })
