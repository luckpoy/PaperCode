# "images"
"images" 的value是一个字典列表，列表每个元素是一个字典，包含图片的信息，例子
```
{"file_name": "training/0336.png", "height": 1200, "width": 1600, "segm_file": "training/0336.xml", "id": 0}
```
* "file_name" 指定文件位置，要注意和前面设置的“img_prefix”目录前缀对应，确保能读取到文件
* "segm_file" 每张图片单独的标注文件，可有可无，实际在"annotations"中定义
* "id" 图片id，很重要，后面的annotation要会对应每个image_id
# "categories"
"categories"的value是一个字典列表，即标签类别，因为我们ocr里面都是文本类别，所以只要一类就行，无脑复制粘贴即可
```
"categories": [{"id": 1, "name": "text"}] 
```
# "annotations"
"annotations" 的value也是一个字典列表，列表每个元素是一个字典，即最终读取的ground truth标签，例子
```
{"iscrowd": 0, "category_id": 1, "bbox": [213, 16, 370, 1163], "area": 168314.0, "segmentation": [[485, 1179, 306, 991, 252, 800, 213, 608, 215, 413, 274, 214, 402, 16, 535, 130, 471, 291, 296, 460, 301, 620, 365, 777, 490, 931, 583, 1089]], "image_id": 0, "id": 0}
```
* "iscrowd", 0是polygon格式segmentation；1是RLE格式segmentation，参考上面coco数据格式
* "category_id" 目标类别，反正都是文本
* "bbox" [x,y,w,h]形式的gt，前两个是左上角点坐标，w h 是框的宽高
* "area" segmentation面积
* "segmentation" [x1,y1,x2,y2...]多边形的gt，每两个是一个点的一对坐标
* "image_id" 对应图片的id，要一一对应哈
* "id" 很重要，每个图片可能有多个目标，此id要全局唯一性，所以取值[0-总segmentation个数]，不能每次遍历一张图片时，id又从0开始