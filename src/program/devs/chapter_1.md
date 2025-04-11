# WSL

## 安装

```
wsl --install
```

## 文件共享

**从 Windows 访问 WSL 文件**

在文件资源管理器地址栏输入
```
\\wsl$\Ubuntu\home\
```

**从 WSL 访问 Windows 文件**

Windows 的盘符会自动挂载在 /mnt/ 下
``` bash
xiaolh@DESKTOP-B3VFKS8:~/work$ ls /mnt/
c  d  wsl  wslg
xiaolh@DESKTOP-B3VFKS8:~/work$
```

## 程序、脚本联动

**在 WSL 中运行 Windows 程序**

``` bash
explorer.exe .        # 打开当前文件夹在 Windows 文件管理器中
code .                # 如果已安装 VSCode，会打开当前目录

xiaolh@DESKTOP-B3VFKS8:~$ which code
/mnt/d/apps/VSCode/bin/code
```
