#TODO make into actual module
try:
  import pyximport; pyximport.install()
  from csgd import StructuredClassifier
except ImportError:
  from sgd import StructuredClassifier
