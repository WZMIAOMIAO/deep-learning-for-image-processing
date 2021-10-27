import json
import numpy as np
from PIL import Image

# 读取mask标签
target = Image.open("./2007_001288.png")
# 获取调色板
palette = target.getpalette()
palette = np.reshape(palette, (-1, 3)).tolist()
# 转换成字典子形式
pd = dict((i, color) for i, color in enumerate(palette))

json_str = json.dumps(pd)
with open("palette.json", "w") as f:
    f.write(json_str)

# target = np.array(target)
# print(target)
