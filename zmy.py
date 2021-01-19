from turtle import *


def curvemove():
    for i in range(100):
        right(2)
        forward(2)


color('red', 'pink')

begin_fill()



left(140)

forward(111.65)

curvemove()

left(120)


curvemove()

forward(111.65)

goto(-30,70)
write("曾梦圆",font=("arial",20,"normal"))


end_fill()

done()
