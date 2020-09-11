def aplusbsq(a,b):
    return  lambda a,b: (a*a + b*b + 2*a*b)   #(a*a + b*b + 2*a*b)
aplusb = aplusbsq(1,2)
def aminusbsq(a,b):
    return lambda a,b: (a*a + b*b - 2*a*b)
aminusb = aminusbsq(1,2)
def aplusminusb(a,b):
    return lambda a,b:((a+b)*(a-b))
apmb = aplusminusb(1,2)