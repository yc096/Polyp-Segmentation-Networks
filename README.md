# Polyp-Segmentation-Networks

- Model
    - [__ENet__](https://github.com/yc096/Polyp-Segmentation-Networks/blob/main/Model/ENet.py)  ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation [_
      Paper_](https://arxiv.org/abs/1606.02147)
    - [__Model 1__](https://github.com/yc096/Polyp-Segmentation-Networks/blob/main/Model/Model1.py)
- Module
    - [__OctaveConv__](https://github.com/yc096/Polyp-Segmentation-Networks/blob/main/Module/OctaveConv.py) Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave
      Convolution [_paper_](https://arxiv.org/abs/1904.05049)
    - [__ChannelAttention__](https://github.com/yc096/Polyp-Segmentation-Networks/blob/main/Module/ChannelAttention.py) Squeeze-and-Excitation Networks [_
      paper_](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf)
    - [__SpatialAttention__](https://github.com/yc096/Polyp-Segmentation-Networks/blob/main/Module/SpatalAttention.py)
    - [__Functional__](https://github.com/yc096/Polyp-Segmentation-Networks/blob/main/Module/.py)
        - Channel Split
        - Channel Shuffle
- [__Loss__](https://github.com/yc096/Polyp-Segmentation-Networks/blob/main/Utils/loss.py)
    - wiou+wbce
    - Dice