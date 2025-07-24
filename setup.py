import setuptools


setuptools.setup(
    name="AE_Toolbox",
    version="0.2.0",
    author="xiaoxinXDU",
    author_email="1429030919@qq.com",
    description="A toolbox for adversarial examples generate",
    py_modules=["AE_Toolbox_3.0.common"],
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent"],
    # 自动找到项目中导入的模块
    packages=setuptools.find_packages(),
    # 依赖模块
    install_requires=['foolbox==3.3.3',
                      'torch==1.13.0',
                      'torchvision==0.14.0',
                      'numba==0.56.4',
                      'pandas==1.5.3'],
    python_requires=">=3.8"
)