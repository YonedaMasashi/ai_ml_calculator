from django import forms

class CalculatorForm(forms.Form):
    first = forms.IntegerField(label='first')
    second = forms.IntegerField(label='second')

