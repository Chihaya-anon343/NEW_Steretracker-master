# C++ 系统开发 Dockerfile 与容器化环境配置教程

在企业级 C++ 开发中，配置 Docker 的核心目标是：**统一编译工具链版本、解决依赖冲突，并确保在容器内生成的文件在宿主机上不会出现权限问题。**

本教程专为结合 CMake、CTest 和 Git 的 C++ 系统开发量身定制，涵盖 Dockerfile 编写、指令解析、镜像构建与挂载运行，以及企业内网环境下的镜像迁移。

---

## 一、 标准化 Dockerfile 示例

请在项目的根目录下创建一个名为 `Dockerfile` 的无后缀文件，并将以下内容复制进去：

```
# 1. 明确基础镜像：使用 LTS 长期支持版，保证底层环境稳定
FROM ubuntu:22.04

# 2. 环境变量配置：避免 apt 安装时出现交互式时区选择卡顿
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 3. 安装 C++ 核心工具链与构建工具 (CMake, Git 等)
# 最佳实践：将 update、install 和清理缓存写在同一个 RUN 指令中，以减小镜像体积
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    gdb \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    ca-certificates \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. 配置非 root 用户（极其关键！）
# 防止容器内编译产生的输出文件在宿主机上变为 root 权限，导致宿主机用户无法修改或删除
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# 5. 切换到普通用户，并设置工作目录
USER $USERNAME
WORKDIR /workspace

# 6. 设置默认启动命令，保持容器交互运行
CMD ["/bin/bash"]
```

---

## 二、 Dockerfile 核心指令解析

* `FROM`：定义镜像底座。避免使用 `latest` 标签，务必使用明确的版本号（如 `ubuntu:22.04`），这能保证未来重新构建时基础环境依然一致。
* `RUN`：镜像构建时执行的命令。每一个 `RUN` 指令都会增加镜像的层数和体积，因此请使用 `&&` 将相关的安装命令串联，并在末尾执行 `rm -rf /var/lib/apt/lists/*` 清理 apt 索引缓存。
* `ARG`** vs **`ENV`：
  * `ARG`：仅在 `docker build` 构建镜像阶段有效（如传递的用户 UID/GID），构建完成后即失效。
  * `ENV`：设置环境变量，在容器运行期间持续生效。
* `USER`：强烈建议在 Dockerfile 末尾将运行身份由 root 切换为非 root 普通用户。这能保证容器内生成的文件与宿主机用户同权，并与 VS Code DevContainers 及 CLion 远程开发模式完美契合。

---

## 三、 镜像构建与运行指南

写好 Dockerfile 后，可以通过以下步骤将其构建为镜像，并启动运行容器。

### 1. 构建镜像 (Build)

在 `Dockerfile` 所在的目录下执行：

```
docker build -t cpp-dev-env:v1 .
```

> **说明** ：`-t` 参数用于指定镜像名称和标签，`.` 代表构建上下文所在目录。

### 2. 运行容器并挂载代码 (Run)

这是日常开发中最常用的一条命令，它将宿主机当前的代码目录挂载至容器内部：

```
docker run -it --rm -v $(pwd):/workspace cpp-dev-env:v1
```

> **参数详解** ：
>
> * `-it`：开启交互式终端并分配伪终端。
> * `--rm`：退出容器后自动清理并销毁容器实例，保持系统干净。
> * `-v $(pwd):/workspace`：将宿主机当前目录映射到容器内的 `/workspace` 目录。

进入容器后，您可以直接运行标准的 CMake 构建与测试流程：

```
# 生成构建文件
cmake -B build -G Ninja

# 编译代码
cmake --build build

# 执行单元测试
ctest --test-dir build --output-on-failure
```

---

## 四、 企业级进阶：内网镜像与迁移

### 1. 替换企业内网镜像源

如果在企业内网环境下无法直接拉取 Docker Hub 官方镜像，请将 Dockerfile 首行的基础镜像修改为公司内部 Artifactory 地址：

```
FROM your-company-artifactory.com/docker/ubuntu:22.04
```

### 2. 使用 Skopeo 进行无 Daemon 镜像迁移

在无 Docker 守护进程的环境或多台私有服务器之间迁移镜像时，推荐使用 **Skopeo** 工具：

* **将本地导出的镜像包推送至内部仓库** ：

```
  skopeo copy docker-archive:cpp-dev-env.tar docker://internal-artifactory.com/repo/cpp-dev-env:v1
```

* **跳过 TLS 证书检查（针对企业内部自签名证书）** ：

```
  skopeo copy --dest-tls-verify=false docker-archive:cpp-dev-env.tar docker://internal-artifactory.com/repo/cpp-dev-env:v1
```
