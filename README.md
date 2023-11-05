# gpu-ray-tracing

这是日常使用时用于调试和测试使用的GPU运行光线追踪的代码。参考文献为：Understanding the Efficiency of Ray Traversal on GPUs和Architecture Considerations for Tracing Incoherent Rays。对于光线追踪硬件加速器的设计皆是基于该算法进行的。


**下面是对应于不同文章和用处对代码进行修改进行的记录**

## IEEE TVLSI

*2023.9.5*

Q1:需要对现有的测试场景增加AO、Diffuse光线类型的测试。  
A1:按照原先测试的方法以及对代码和项目的逐步理解，有两种方式进行实现：  
(1)对trace函数的光线进行printf然后重定向到某个文件中；  
(2)使用cuda的内存模型来进行数据复制。

实现预计与之前的方法类似，但是可以借助anyhit作为表示来选择输出，另外，最好利用cuda内存模型输出。

Q2:测试场景比较少，需要增加测试场景的数量。  
A2:在review推荐的文章中有一篇可以用来参考的测试场景集合的obj格式可以下载，但是目前由于camera不合适等各种原因，还未能实现。需要写出类似于camera的反函数来进行调整相机位置。  


*2023.9.6*

**http://www.styb.cn/cms/ieee_754.php**  这个是日常用于转换IEEE 754和单精度浮点数的网页工具


*2023.9.7*

发现CameraControls.cc中包括encodeSignature的函数，所以考虑从这个来当作decode的反函数，可能与blender之类的程序有一定的联系。

*2023.9.11*

1. 通过不断测试寻找到了适合bunny和dragon的camera；  
2. 通过对cuda的学习，找到了搬运数据的函数。  
光线： **copyRays**  
BVH:  **cudaBVH**  
三角形：**CudaBVH**  

*2023.9.12*

1. 使用fprintf()函数可以写入文件中;  
2. 函数返回指针和CUDA返回数据测试。  
上述代码均上传至github: https://github.com/yanrun000/ray_tracing_testcode.gi

*2023.9.15-9.20记录实验结果*

bunny   internal nodes  50876,三角形个数144500。

dragon  internal nodes  301376,三角形个数910348。(1.1 M)

conference  internal nodes 105025,三角形个数350949   

hairball  internal nodes 1249052,三角形个数6469561(6.9 M)  

sponza  internal nodes 35907,三角形个数121384

Mori Knob  internal nodes   3483，三角形个数12570

## Primary  
sponza  *597.51 Mray/s*      IST 46.97% TRV1 99.99% TRV2 99.99%    
Mori Knob  *1271.61 Mray/s*  IST 99.83%  TRV1 87.53% TRV2 87.54%  
hairball *280.49 Mray/s*     IST 10.63% TRV1 45.16% TRV2 34.75%  
Dragon  *575.43 Mray/s*      IST 12.72%  TRV1 41.11% TRV2 42.76%  
Bunny  *825.11 Mray/s*       IST 12.67%  TRV1 42.51%  TRV2 39.04%

## Diffuse
conference *831.28 Mray/s*  IST 73.14% TRV1 99.73% TRV2 99.38%  
fairy   *678.77 Mray/s* IST 14.28% TRV1 96.42% TRV2 96.36%   
sibenik  *286.97 Mray/s*    IST 24.59%  TRV1 62.72% TRV2 61.42%   
san      *132.28 Mray/s*  IST 38.99%  TRV1 99.99% TRV2 99.99%   
sponza   *325.33 Mray/s*  IST 50.35%  TRV1 99.99% TRV2 99.99%   
Mori Knob  *1466.05 Mray/s*  IST 33.49%  TRV1 99.99% TRV2 99.99%

## AO
conference  *1478.43 Mray/s*  IST 50.09% TRV1 99.99% TRV2 99.99%  
fairy   *1280.77 Mray/s* IST 83.56%  TRV1 99.99% TRV2 99.99%  
sibenik   *1499.86 Mray/s* IST 29.88%  TRV1 99.99% TRV2 99.99%  
san        *556.89 Mray/s*  IST 10.91% TRV1 99.99% TRV2 99.99%  
sponza   *1022.61 Mray/s*   IST 35.55% TRV1 99.99% TRV2 99.99%  
Mori Knob *2763.01 Mray/s*  IST 15.56%  TRV1 99.99% TRV2 99.99%  

