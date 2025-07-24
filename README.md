# AADF(opensource)
Adversarial Attack and Defense Framework for Physical Anti-interference Technology in Real World(60+ methods)

# Framework video demonstration
https://github.com/user-attachments/assets/008895d6-a105-44aa-be74-2f14252f1bb3

# pycharm Terminal usage method
python verify.py --model model-name --vnnlib constraints-name

example 1 use VGG16 model
（enter NNverify/complete_verifier/）
python verify.py --model vgg16 --vnnlib spec0_screw.vnnlib
vgg16 model file is in NNverify/complete_verifier/vnncomp2022_benchmarks/benchmarks/vggnet16/onnx/

example 2 use MNIST FC model
（enter NNverify/complete_verifier/）
python verify.py --model mnistfc --vnnlib prop_0_0.03.vnnlib
mnistfc model file is in NNverify/complete_verifier/vnncomp2022_benchmarks/benchmarks/mnistfc/onnx/

example 3 use ResNet model
（enter NNverify/complete_verifier/）
python verify.py --model resnet --vnnlib cifar10_spec_idx_1551_eps_0.00198.vnnlib
mnistfc model file is in NNverify/complete_verifier/vnncomp2022_benchmarks/benchmarks/sri_resnet_a/onnx/

# front-end use method
1. download the whole project file
2. enter AADF/src
3. terminal enter "python AE_front0318.py"

