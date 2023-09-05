# gpu-ray-tracing

这是日常使用时用于调试和测试使用的GPU运行光线追踪的代码。参考文献为：Understanding the Efficiency of Ray Traversal on GPUs和Architecture Considerations for Tracing Incoherent Rays。对于光线追踪硬件加速器的设计皆是基于该算法进行的。


**下面是对应于不同文章和用处对代码进行修改进行的记录**

## IEEE TVLSI

*2023.9.5*

Q1:需要对现有的测试场景增加AO、Diffuse光线类型的测试。  
A1:按照原先测试的方法以及对代码和项目的逐步理解，有两种方式进行实现：  
(1)对trace函数的光线进行printf然后重定向到某个文件中；  
(2)使用cuda的内存模型来进行数据复制。

Q2:测试场景比较少，需要增加测试场景的数量。
A2:在review推荐的文章中有一篇可以用来参考的测试场景集合的obj格式可以下载，但是目前由于camera不合适等各种原因，还未能实现。需要写出类似于camera的反函数来进行调整相机位置。

