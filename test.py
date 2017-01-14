import sys
from caption import *
import urllib, cStringIO
from PIL import Image

model = Caption_Generator(mode = 'test')
file = cStringIO.StringIO(urllib.urlopen(sys.argv[1]).read())
img = Image.open(file)
model.decode(img)
