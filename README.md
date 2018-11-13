# Face detection and alignment by mtcnn
Using the pre-model of the trained MTCNN, face detection and  alignment are performed on the input image, and the output is the aligned face image(112 * 112).

* The model and detection code in `FaceDetection` directory

* The main code: `detect.py`: The code includs image augmentation operations
  ```shell
  python detect.py -i ./testset/(your images path) -o ./testset_aligned -j 8 -r 0.25,0.5,1(-r is for image augmentation)
  ```
