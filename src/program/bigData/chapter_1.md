# hadoop

Hadoop就是大数据技术的一种实现方式。

Hadoop是一些列技术，包括 HDFS（存储），MaqReduce（算法），YARN（计算任务调度）。

一些其他的组件，包括 Hive可以把SQL语句编译为MaqReduce，Kafka作为一种存储任务的队列，防止存储任务太多，扛不住。

SPark是替换MaqReduce的算法组件，SparkSQL可以把SQL语句编译为Spark，MLlib是基于Spark做机器学习的，

Zookeeper是一个分布式资源管理组件，管理硬件的兼容。

Azkaban是一个任务流调度软件。

## 搭建测试环境
``` bash
docker run -it --name hadoop -p 9870:9870 -p 9864:9864 -p 8088:8088 bigdataeurope/hadoop:3.3.1 

$HADOOP_HOME/sbin/start-dfs.sh
$HADOOP_HOME/sbin/start-yarn.sh
```

## HDFS
HDFS（Hadoop Distributed File System） 是一个分布式文件系统。

master/slave，active/standby。

以block的方式将文件进行拆分，并存储在不同节点上。

### 安装

### 使用
命令行: 前缀 + 命令

和ftp差不多
```
hdfs fs [-get] [-put] [ls]
```

远程客户端，可以配置HDFS服务端的URL（hdfs://nameserver/data_path）

API接口
``` bash 
# 创建空文件
PUT http://<namenode>:9870/webhdfs/v1/user/hadoop/test.txt?op=CREATE 

# 上传文件
PUT http://<datanode>:9870/webhdfs/v1/user/hadoop/test.txt?op=CREATE&data=true [Content-Type: application/octet-stream]

# 读取文件
GET http://<namenode>:9870/webhdfs/v1/user/hadoop/test.txt?op=OPEN

# 获取文件状态
GET http://<namenode>:9870/webhdfs/v1/user/hadoop/test.txt?op=GETFILESTATUS

```

