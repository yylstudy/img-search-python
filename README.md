图片搜索服务python源码，基于
https://github.com/milvus-io/bootcamp/tree/master/solutions/image/reverse_image_search/quick_deploy

https://github.com/milvus-io/bootcamp/issues/1140
改造，这里img-search-server的原镜像是由python实现，且所有方法都加上了async，也就是单线程方法 ，所以需要去除所有的async

改造后代码见

