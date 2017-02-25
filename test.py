import sys
from caption import *
import urllib, cStringIO
from PIL import Image
img_path='Images/test.png'
model = Caption_Generator(mode = 'test')
model.decode(sys.argv[1])
