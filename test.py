import sys
from caption import *
import urllib, cStringIO
from PIL import Image
img_path='Images/test.png'
model = Caption_Generator(mode = 'test')
file = cStringIO.StringIO(urllib.urlopen(sys.argv[1]).read())
img = Image.open(file)
img.save(img_path)
decode_graph = model.build_decode_graph()
model.decode(decode_graph, img_path)
