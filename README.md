DensePoint
===
This repository contains the code in Pytorch for the paper:

__DensePoint: Learning Densely Contextual Representation for Efficient Point Cloud Processing__ [[arXiv](https://arxiv.org/abs/1909.03669)] [[CVF](http://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_DensePoint_Learning_Densely_Contextual_Representation_for_Efficient_Point_Cloud_Processing_ICCV_2019_paper.pdf)]
<br>
[Yongcheng Liu](https://yochengliu.github.io/), [Bin Fan](http://www.nlpr.ia.ac.cn/fanbin/), [Gaofeng Meng](http://www.escience.cn/people/menggaofeng/index.html;jsessionid=EE2E193290F516D1BA8E2E35A09A9A08-n1), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/), [Shiming Xiang](https://scholar.google.com/citations?user=0ggsACEAAAAJ&hl=zh-CN) and [Chunhong Pan](http://people.ucas.ac.cn/~0005314)
<br>
[__ICCV 2019__](http://iccv2019.thecvf.com/)

## Citation

If our paper is helpful for your research, please consider citing:   

        @inproceedings{liu2019densepoint,   
            author = {Yongcheng Liu and    
                            Bin Fan and  
                       Gaofeng Meng and
                           Jiwen Lu and
                      Shiming Xiang and   
                           Chunhong Pan},   
            title = {DensePoint: Learning Densely Contextual Representation for Efficient Point Cloud Processing},   
            booktitle = {IEEE International Conference on Computer Vision (ICCV)},    
            pages = {5239--5248},  
            year = {2019}   
        }   

## Usage: Preparation

- Requirement

  - Ubuntu 14.04
  - Python 3 (recommend Anaconda3)
  - Pytorch 0.3.\*
  - CMake > 2.8
  - CUDA 8.0 + cuDNN 5.1

- Building Kernel

      git clone https://github.com/Yochengliu/DensePoint.git 
      cd DensePoint
      mkdir build && cd build
      cmake .. && make

- Dataset
  - Shape Classification: download and unzip [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) (415M). Replace `$data_root$` in `cfgs/config_cls.yaml` with the dataset parent path.

## Usage: Training
- Shape Classification

      sh train_cls.sh
        
We have trained a 6-layer classification model in `cls` folder, whose accuracy is 92.38%.

## Usage: Evaluation
- Shape Classification

      Voting script: voting_evaluate_cls.py
        
You can use our model `cls/model_cls_L6_iter_36567_acc_0.923825.pth` as the checkpoint in `config_cls.yaml`, and after this voting you will get an accuracy of 92.5% if all things go right.

## License

The code is released under MIT License (see LICENSE file for details).

## Acknowledgement

The code is heavily borrowed from [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch).
        
## Contact

If you have some ideas or questions about our research to share with us, please contact <yongcheng.liu@nlpr.ia.ac.cn>
