"""
## Test
"""
from math import log2
vocabulary = ' !"\'(),-.0123456789:;?abcdefghijklmnopqrstuvwxyz'

# returns a probability in (0,1)
# return values must sum to 1.0 over all possible characters c
def anything_goes_model(c, history):
  letter_mass = 0.8
  if c >= 'a' and c <= 'z':
    return letter_mass/26.0
  else:
    return (1-letter_mass)/(len(vocabulary)-26)
def evaluate(lang):
  testfile = open(lang+'-test.txt', 'r')
  max_history = 100
  history = []
  loss_anything_goes = 0
  count = 0
  while True:
    c = testfile.read(1)
    if not c:
      break
    count += 1
    loss_anything_goes -= log2(anything_goes_model(c, history))
    if len(history) == max_history:
      history.pop(0)
    history.append(c)
  return loss_anything_goes/count

   