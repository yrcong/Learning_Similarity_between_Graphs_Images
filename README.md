# Learning Similarity between Scene Graphs and Images
PyTorch Implementation of the Paper [**Learning Similarity between Scene Graphs and Images with Transformers**](https://arxiv.org/abs/2304.00590).

The project page is [**AVAILABLE**](https://yrcong.github.io/gicon/)!

Now it is just the draft version. You can use the code to evaluate your SGG model:D

# 1. Installation
Download **GICON Repo** with:
```
git clone https://github.com/yrcong/Learning_Similarity_between_Graphs_Images.git
```

To run the code, we use 
```
python==3.7
pytorch==1.11.0
torchvision==0.12.0 
```
# 2. Data Preparation for Visual Genome:
a) Download the the images of Visual Genome [Part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) and [Part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Unzip and place all images in a folder ```data/vg/images/```

b) Download the annotations of [VG (in COCO-format)](https://drive.google.com/file/d/1aGwEu392DiECGdvwaYr-LgqGLmWhn8yD/view?usp=sharing) and unzip it in the ```data/``` forder.

c) Download the pretrained model [location_bound](https://cloud.tnt.uni-hannover.de/index.php/s/WXkN8Nf4R4mpmTe) and [location_free](https://cloud.tnt.uni-hannover.de/index.php/s/296NnSaEzxaPa32) and put the checkpoints under ```ckpt/```.

# 3. TODO: OpenImages
# 4. Benchmarking
a) Create a pickle file containing the predictions of the scene graph generation model. We provide a template ([BGNN prediction PKL file](https://cloud.tnt.uni-hannover.de/index.php/s/w3jeKgJg62g8e5W)), 
You can download it and save it under the working path.

please check the data format and ensure your prediction file is consistent with this template!

To compute R-Precision for Locaiton-Free Graphs (K=100):
```
python benchmark.py --batch_size 100 --image_layer_num 6 --graph_layer_num 6 --resume ckpt/location_free.pth --eval --prediction bgnn_prediction.pkl
```

To compute R-Precision for Locaiton-Bound Graphs (K=100):
```
python benchmark.py --batch_size 100 --image_layer_num 6 --graph_layer_num 6 --resume ckpt/location_bound.pth --eval --prediction bgnn_prediction.pkl --node_bbox
```

# 5. TODO: Training
# 6. TODO: Evaluation

