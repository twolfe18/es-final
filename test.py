
from training import *

ft = FixedAdditiveTrainer()

print 'simple'
print ft.get_data(onlyTake=10)

print 'reservoir'
print ft.get_data(onlyTake=10, reservoir=True)

