# 2w5验证集预处理

验证集的60%部分下载地址：

链接:https://pan.baidu.com/s/1VyN8mUgIHwBmb7rjy91vxA  密码:i2mw

* `detect.py`

  > pyseeta检测人脸，剪裁并保存

  ```shell
  python3 ../val_data_preprocess/detect.py -i ./testset/ -o ./testset_aligned -j 8 -r 0.25,0.5,1
  ```

* `add_occlusion.py`

  > 添加occlusion小黑块

  ```shell
  python3 ../val_data_preprocess/add_occlusion.py -d ./testset_aligned/ -j 8
  ```

* `preprocess.py`

  > 读取目标文件夹中的人脸图像，切分并保存为若干个MXnet Records

  ```shell
  python3 ../val_data_preprocess/preprocess.py -i ./no_alignment/testset_detected/ -t ./emotioNet_challenge_files_server_challenge_1.2/dataFile_1000.txt -rec ./testset_records/augmented/testset.rec -sp 0.6,0.3
  
  python3 ../val_data_preprocess/preprocess.py -i ./testset_aligned/ -t ./dataFile_clean_val.txt -rec ./testset_records/original/testset.rec -sp 1
  ```
