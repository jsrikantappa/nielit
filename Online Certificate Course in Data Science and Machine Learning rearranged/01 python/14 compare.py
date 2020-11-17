a=input()
int(a)
b=input()
int(b)
big=lambda a,b: max(a,b)
small=lambda a,b: min(a,b)
equal=lambda a,b: 'Both values are equal' if(a==b) else 'Not equal'
print('Biggest is '+big(a,b),'Smallest is '+small(a,b), 'and number are not equal '+equal(a,b))

'''
a=[input(),input()]
big=lambda a: max(a)
small=lambda a: min(a)
equal=lambda a: a+['Both values are equal'] if(a[0]==a[1]) else 'Not equal'
print(big(a),small(a), equal(a))
'''