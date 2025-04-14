# spark

spark 对 python 的支持更好。

## 搭建试验环境

通过 Dcoker 安装 Spark，拉取**bitnami/spark**
``` bash
docker pull bitnami/spark:latest
```

通过 docker-compose 配置集群，保存 `spark/docker-compose.yml`

``` yml
services:
  spark:
    image: docker.io/bitnami/spark
    environment:
      - SPARK_MODE=master
      - SPARK_RPC_AUTHENTICATION_ENABLED=no # 禁用了 RPC 认证、加密和 SSL，以简化配置
      - SPARK_RPC_ENCRYPTION_ENABLED=no
      - SPARK_LOCAL_STORAGE_ENCRYPTION_ENABLED=no
      - SPARK_SSL_ENABLED=no
      - SPARK_USER=spark # 运行用户
    ports:
      - '8080:8080'
  spark-worker:
    image: docker.io/bitnami/spark
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark:7077 # 通过 SPARK_MASTER_URL=spark://spark:7077 连接到主节点
      - SPARK_WORKER_MEMORY=1G # 分配 1GB 内存
      - SPARK_WORKER_CORES=1 # 分配 1 个核心。
      - SPARK_RPC_AUTHENTICATION_ENABLED=no 
      - SPARK_RPC_ENCRYPTION_ENABLED=no
      - SPARK_LOCAL_STORAGE_ENCRYPTION_ENABLED=no
      - SPARK_SSL_ENABLED=no
      - SPARK_USER=spark
```

运行集群
``` bash 
cd spark
docker-compose up --scale spark-worker=3 -d
docker-compose logs spark
docker-compose stop spark
```

提交测试程序
``` bash
docker exec -it 06e7aa696ece /bin/bash
alias ls='ls --color'
spark-submit examples/src/main/python/pi.py
```

**案例程序**
``` python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SparkExample").getOrCreate()

data = [1, 2, 3, 4, 5]
rdd = spark.sparkContext.parallelize(data)

# 显示 RDD 中的数据
# print(rdd.collect())

data = [("Alice", 25), ("Bob", 30), ("Cathy", 28)]
columns = ["Name", "Age"]
df = spark.createDataFrame(data, columns)
df.createOrReplaceTempView("people")

# 执行 SQL 查询
result = spark.sql("SELECT * FROM people WHERE Age > 25")

# 显示查询结果
result.show()
```
