import sys
from lang_map import NLLB_LANGMAP

inlang = sys.argv[1]
outlang = NLLB_LANGMAP[inlang]
print(outlang) 