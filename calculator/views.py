from django.shortcuts import render
from django.http import HttpResponse
from .forms import CalculatorForm
from calculator.learning.calc_ml import *

# Create your views here.
def index(request):
    params = {
        'form' : CalculatorForm()
    }
    if (request.method == 'POST'):
        # learningML()

        first = int(request.POST['first'])
        second = int(request.POST['second'])
        predict_answer = predict(first, second)
        calc_answer = first + second
        params['message'] = 'first:' + request.POST['first'] + \
            '<br>second:' + request.POST['second'] + \
            '<br>calculate answer:' + str(calc_answer) + \
            '<br>predict answer:' + str(predict_answer)
        params['form'] = CalculatorForm(request.POST)
    return render(request, 'calculator/index.html', params)
