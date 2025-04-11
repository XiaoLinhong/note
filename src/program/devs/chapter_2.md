# docker

## 安装

**在 window 中安装 docker** 推荐方案使用 Docker Desktop。

Docker Desktop 实际上在后台启动一个 Linux 虚拟机（WSL2 后端），
你可以在其中运行 Docker 容器，而不需要再手动管理 daemon（Docker的服务）。

[下载安装包](https://docs.docker.com/desktop/setup/install/windows-install/)，
然后在安装的时候选择 **WSL 2 backend**


## docker中的核心概念

- **镜像**: 就像可执行程序（运行某个程序所需的所有内容，包括系统库）
- **容器**：就行进程，是一个运行的镜像，Image+运行时状态
- **网络**：Docker 默认会为容器分配虚拟网络，使其相互隔离
- **数据卷**: Docker 提供的持久化存储机制，可以把数据挂载在容器外部
- **Dockerfile**: 定义镜像构建过程的脚本（可以从一个基础镜像开始）
- **Docker 引擎**：一个运行时，容器的调度器，整个容器运行的“操作系统”

## 常用命令

## 镜像相关

| 命令 | 说明 |
|------|------|
| `docker images` | 列出本地镜像 |
| `docker pull <镜像名>` | 从 Docker Hub 拉取镜像 |
| `docker build -t <自定义镜像名> .` | 根据 Dockerfile 构建镜像 |
| `docker rmi <镜像ID或名称>` | 删除镜像（不能有依赖容器） |
| `docker tag <镜像ID> <新名称>` | 重命名镜像 |


### 容器相关

| 命令 | 说明 |
|------|------|
| `docker ps` | 查看正在运行的容器 |
| `docker ps -a` | 查看所有容器（包括已停止） |
| `docker run <镜像名>` | 运行一个容器 |
| `docker run -it <镜像名> /bin/bash` | 交互式运行并进入容器 |
| `docker exec -it <容器名或ID> /bin/bash` | 进入已经运行的容器 |
| `docker stop <容器ID或名称>` | 停止容器 |
| `docker start <容器ID或名称>` | 启动已停止的容器 |
| `docker restart <容器ID或名称>` | 重启容器 |
| `docker rm <容器ID或名称>` | 删除容器（需先停止） |
| `docker logs <容器ID>` | 查看容器输出日志 |

## 网络与数据卷

| 命令 | 说明 |
|------|------|
| `docker network ls` | 列出网络 |
| `docker network create <名称>` | 创建网络 |
| `docker volume ls` | 列出数据卷 |
| `docker volume create <名称>` | 创建数据卷 |
| `docker run -v <本地路径>:<容器路径>` | 挂载本地目录到容器 |

## 清理命令

| 命令 | 说明 |
|------|------|
| `docker system prune` | 清理无用数据（停止容器、悬挂镜像等） |
| `docker container prune` | 清理所有停止的容器 |
| `docker image prune` | 清理未使用的镜像 |
| `docker volume prune` | 清理未使用的卷 |

## 其他实用命令

| 命令 | 说明 |
|------|------|
| `docker inspect <容器或镜像>` | 查看详细信息（JSON） |
| `docker cp <容器>:<路径> <本地路径>` | 从容器拷贝文件到本地 |
| `docker stats` | 查看容器资源占用情况 |

