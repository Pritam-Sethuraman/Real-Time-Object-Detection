# Real Time Object Detection

<p>
SSD (Single Shot MultiBox Detector) is a popular algorithm in object detection. It has no delegated region proposal network and predicts the boundary boxes and the classes directly from feature maps in one single pass. improve accuracy, SSD introduces: small convolutional filters to predict object classes and offsets to default boundary boxes. Mobilenet is a convolution neural network used to produce high-level features. SSD is designed for object detection in real-time
The SSD object detection composes of 2 parts: Extract feature maps, and apply convolution filters to detect objects
  
Arguments used here:
- prototxt = MobileNetSSD_deploy.prototxt.txt (required)
- model = MobileNetSSD_deploy.caffemodel (required)
- confidence = 0.2 (default)
  
OpenCV provides two functions to facilitate image preprocessing for deep learning classification: cv2.dnn.blobFromImage and cv2.dnn.blobFromImages. Here we will use cv2.dnn.blobFromImage. These two functions perform: Mean subtraction, Scaling, and optionally channel swapping

Mean subtraction is used to help combat illumination changes in the input images in our dataset. We can therefore view mean subtraction as a technique used to aid our Convolutional Neural Networks. Before we even begin training our deep neural network, we first compute the average pixel intensity across all images in the training set for each of the Red, Green, and Blue channels. We end up with three variables: mu_R, mu_G, and mu_B (3-tuple consisting of the mean of the Red, Green, and Blue channels)

For example, the mean values for the ImageNet training set are R=103.93, G=116.77, and B=123.68
When we are ready to pass an image through our network (whether for training or testing), we subtract the mean, \mu, from each input channel of the input image:
- R = R - mu_R
- G = G - mu_G
- B = B - mu_B

We may also have a scaling factor, \sigma, which adds in a normalization:
- R = (R - mu_R) / sigma
- G = (G - mu_G) / sigma
- B = (B - mu_B) / sigma

The value of \sigma may be the standard deviation across the training set (thereby turning the preprocessing step into a standard score/z-score). Sigma may also be manually set (versus calculated) to scale the input image space into a particular range — it really depends on the architecture, how the network was trained

cv2.dnn.blobFromImage creates 4-dimensional blob from image. Optionally resizes and crops image from center, subtract mean values, scales values by scalefactor, swap Blue and Red channels. A blob is just an image(s) with the same spatial dimensions (width and height), same depth (number of channels), that have all be preprocessed in the same manner
</p>
  
---

## Requirements
- Python3
- OpenCV

---

## How to Use
- Clone the repository
```html
git clone https://github.com/Pritam-Sethuraman/Real-Time-Object-Detection.git
```

- Create a Virtual Environment
```html
python -m venv Real_Time_Object_Detection
```


- Install the required dependancies
```html
pip install -y -r requirements.txt
```

---
## Created & Maintained By

[Pritam Sethuraman](https://github.com/pritam-sethuraman)

> If you found this project helpful or you learned something from the source code and want to thank me, consider buying me a cup of :coffee:
>
> - [PayPal](https://paypal.me/pritam2500/)


---

## License
MIT License

Copyright (c) 2022 Pritam Sethuraman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---
