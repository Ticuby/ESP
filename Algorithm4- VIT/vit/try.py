import cv2

# x = cv2.imread('.\\images\\Base11\\20051019_38557_0100_PP.tif')
# x_hui = cv2.imread('.\\images\\Base11\\20051019_38557_0100_PP.tif',0)
# crop = int((x.shape[1] - x.shape[0]) / 2)
# x = x[0:x.shape[0], crop:crop + x.shape[0]]
# x_hui = x_hui[0:x_hui.shape[0], crop:crop + x_hui.shape[0]]
# x_hui = cv2.resize(x_hui,(512,512))
# x = cv2.resize(x,(512,512))
#
# _, green, _ = cv2.split(x)
# clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16))  # 自适应直方图均衡化
# green_clahe = clahe.apply(green)
# green_clahe_median = cv2.merge((green_clahe,green_clahe,green_clahe))
# green_clahe_median_blur = cv2.medianBlur(green_clahe_median, 3)
#
#
# cv2.imshow('green_clahe_median_blur', green_clahe_median_blur)
# cv2.imshow('green_clahe_median', green_clahe_median)
# cv2.imshow('green_clahe',green_clahe)
# cv2.imshow('green',green)
# cv2.imshow('x',x)
# cv2.imshow('x_hui',x_hui)
# cv2.waitKey(0)

a =[[142,	6,	14,	 4],
[34,	2,	7,	 1],
[29,	1,	26,	 11],
[6,	0,	11,	 66]]




x = 0
y = 0
for i in range(1,4):
    for j in range(1,4):
        x += a[i][j]
    y += a[i][0]
y += x
recall = x/y
print(recall)

y = x
for i in range(1,4):
    y += a[0][i]
precision = x/y
print(precision)


