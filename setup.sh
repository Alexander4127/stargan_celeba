pip install -r requirements.txt

git clone https://github.com/S-aiueo32/lpips-pytorch.git
mv lpips-pytorch/lpips_pytorch .

mkdir -p data/celeba
wget https://raw.githubusercontent.com/taki0112/StarGAN-Tensorflow/master/dataset/celebA/list_attr_celeba.txt -O data/celeba/tmp.txt
tail -n +2 data/celeba/tmp.txt > data/celeba/list_attr_celeba.txt
