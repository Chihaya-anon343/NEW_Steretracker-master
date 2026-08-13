
# Eclipse Capella 开源 MBSE 建模工具快速上手指南

Eclipse Capella 是由法国 Thales（泰雷兹）集团开源、目前在工业界应用最广泛的基于模型的系统工程（MBSE）建模软件之一。与传统通用 UML/SysML 工具不同，Capella  **原生集成了 Arcadia 系统工程方法论** ，提供了从业务需求分析到物理架构落地的高度结构化建模流程。

本文将带领您从零开始，快速掌握 Capella 的安装配置、核心概念、Arcadia 四层建模实战以及高级进阶技巧。

---

## 一、 Capella 与 Arcadia 方法论核心理念

在使用 Capella 之前，必须理解其背后的核心方法论—— **Arcadia（Architecture Analysis and Design Integrated Approach）** 。Arcadia 将系统建模严格划分为递进的  **4 个视图视角（Perspectives）** ：

```
[1. 业务概念视角] Operational Analysis (OA)   --> 用户需要做什么？(Operational Capabilities/Activities)
       │
       ▼
[2. 系统功能视角] System Analysis (SA)        --> 系统必须完成什么？(System Functions/Boundary)
       │
       ▼
[3. 逻辑结构视角] Logical Architecture (LA)   --> 系统如何按模块划分？(Logical Components/Data Flow)
       │
       ▼
[4. 物理实现视角] Physical Architecture (PA)  --> 系统具体如何构建与实现？(Hardware/Software/Nodes)
```

null.  **Operational Analysis (OA) - 业务 operational 分析** ：关注 **用户与利益相关者** （Operational Entities/Actors）的实际业务场景，不涉及技术或系统本身。
null.  **System Analysis (SA) - 系统需求分析** ：将系统视为一个 **黑盒（Black Box）** ，明确系统边界、外部接口、系统级功能与交互流。
null.  **Logical Architecture (LA) - 逻辑架构设计** ：将系统拆解为 **白盒（White Box）** ，定义逻辑组件（Logical Components），并将功能分配（Allocate）给逻辑组件。
null.  **Physical Architecture (PA) - 物理架构设计** ：明确硬件节点（Node PC）、软件行为组件（Behavior PC）、物理接口（Physical Link/Port）及具体代码/硬件落地。

---

## 二、 环境准备与安装

### 1. 下载与安装

* **下载地址** ：访问 [Eclipse Capella 官网](https://www.eclipse.org/capella/download.html)。
* **解压即用** ：Capella 为绿色免安装版，下载压缩包（如 `capella-6.x.x-win32-x86_64.zip`）后解压至无中文与空格的路径下（例如 `D:\Tools\Capella`）。
* **Java 环境** ：Capella 较新版本已嵌入微型 JRE 环境，无需额外安装 Java；若版本要求单独 JRE，请确保本地已配置 JDK 17+。

### 2. 推荐常用插件（Add-ons）

在实际项目中，建议在 Capella 的 `Dropins` 或通过菜单 `Help -> Install New Software` 配置以下插件：

* **M2Doc** ：基于 Word 模板自动化生成系统架构设计文档（SAD）。
* **Python4Capella** ：允许编写 Python 脚本批量读取/修改 Capella 模型或导出数据。
* **Requirements Viewpoint** ：在模型中直接导入与关联 ReqIF 格式的需求条目。

---

## 三、 Capella 界面布局与导航

启动 Capella 并指定 Workspace（工作空间）后，您将看到标准界面布局：

```
+-----------------------------------------------------------------------------------+
| 菜单栏 / 工具栏                                                                    |
+---------------------+---------------------------------------+---------------------+
|                     |                                       |                     |
|  Project Explorer   |           Diagram Editor              |  Activity Explorer  |
|   (模型元素树)       |            (图形建模区)               |   (Arcadia 流程图)  |
|                     |                                       |   *推荐引导看板*    |
|                     |                                       |                     |
+---------------------+---------------------------------------+---------------------+
|                                 Properties View (属性面板)                         |
+-----------------------------------------------------------------------------------+
```

* **Activity Explorer（活动探索器）** ： **入门最重要的面板** ！它列出了 Arcadia 的各个阶段与推荐的操作步骤，点击按钮即可直接创建对应的图表和元素。
* **Project Explorer（项目浏览器）** ：展示工程树结构，主模型文件后缀为 `.melodymodeller`。
* **Diagram Editor（图表编辑器）** ：绘图区，右侧面板提供绘图工具箱（Palette）。
* **Properties View（属性视图）** ：选中任何模型元素时，在此修改名称、描述、端口和分配关系。

---

## 四、 实战演练：智能温控系统建模

下面以一个简单的“ **智能温控系统（Smart Temp Controller）** ”为例，演示如何通过 Arcadia 四步法在 Capella 中建构模型。

### 0. 新建 Capella 工程

null. 菜单栏选择 `File -> New -> Capella Project`。
null. 输入项目名称（如 `SmartTempController`），点击 `Finish`。
null. 双击打开 `Activity Explorer`。

---

### Step 1: Operational Analysis (OA) - 业务场景分析

明确使用系统的“人和环境”及其目标。

null. 在 Activity Explorer 中展开 **Operational Analysis** 选项卡。
null. 点击  **Create a new Operational Capabilities Diagram (OCB)** 。
null.  **添加实体与参与者** ：
      * 创建 `Operational Entity`：`Room Space`（房间空间）。
      * 创建 `Operational Actor`：`User`（房间住户）。
null.  **定义业务活动（Operational Activity）** ：
      * 创建活动 `Monitor Room Temperature`（监控房间温度）和 `Adjust HVAC Output`（调节空调输出）。
null.  **定义 Operational Process** ：连接活动形成业务流。

---

### Step 2: System Analysis (SA) - 系统黑盒分析

将“智能温控系统”作为一个整体黑盒进行系统分析。

null. 在 Activity Explorer 中展开 **System Analysis** 选项卡。
null.  **实现业务过渡（Transition）** ：
      * 点击 `Perform Automated transition from Operational Analysis`，自动将 OA 阶段的实体与活动平滑过渡为 SA 阶的功能与参与者。
null.  **创建系统架构图 System Architecture Blank (SAB)** ：
      * 界面会自动生成代表整个系统的黑盒 `System`（例如 `Smart Temp Controller System`）和外部 Actor（`User`、`Environment`）。
null.  **定义系统功能与数据流（System Functions & Exchanges）** ：
      * 在系统内部放置功能：`Acquire Temperature Data`（采集温度数据）、`Compute Control Command`（计算控制指令）。
      * 在外部 Actor 上放置功能：`Provide Target Temperature`（设定目标温度）。
      * 连接功能形成  **Functional Exchange** （功能交互流），并在属性面板定义传递的数据类型（Data Items，如 `TempReading`）。

---

### Step 3: Logical Architecture (LA) - 逻辑白盒分解

打破系统黑盒，设计系统内部的逻辑组件划分。

null. 在 Activity Explorer 中切换至  **Logical Architecture** 。
null. 执行 `Automated Transition from System Analysis`，将 SA 阶段的功能过渡到 LA。
null.  **创建逻辑架构图 Logical Architecture Blank (LAB)** 。
null.  **创建逻辑组件（Logical Component）** ：
      * `Sensor Manager`（传感器管理模块）
      * `Control Processor`（控制处理核心）
      * `Actuator Driver`（执行器驱动模块）
null.  **分配功能（Allocate Functions）** ：
      * 将 `Acquire Temperature Data` 拖拽/分配给 `Sensor Manager`。
      * 将 `Compute Control Command` 拖拽/分配给 `Control Processor`。
null.  **创建逻辑接口（Component Exchange）** ：
      * 系统会根据功能交互自动推荐或允许连线创建组件间的数据总线或逻辑端口。

---

### Step 4: Physical Architecture (PA) - 物理/软硬件实现

将逻辑模块落实到具体的硬件节点和软件组件上。

null. 在 Activity Explorer 中切换至  **Physical Architecture** 。
null.  **创建物理架构图 Physical Architecture Blank (PAB)** 。
null.  **定义物理组件分类** ：
      *  **Node Physical Component (Node PC)** ：代表硬件硬件实体，例如 `MCU Microcontroller`、`DS18B20 Temp Sensor Hardware`、`Relay Board`。
      *  **Behavior Physical Component (Behavior PC)** ：代表软件进程/应用，例如 `Control Task App (.elf)`。
null.  **将软件部署至硬件** ：
      * 将 Behavior PC（软件应用）嵌入或部署（Deploy）至对应的 Node PC（MCU 芯片）中。
null.  **定义物理连接（Physical Link）** ：
      * 用 `Physical Link` 连接 `MCU` 与 `Sensor Hardware`，指定物理总线类型（如 `I2C Bus`、`GPIO Pin`）。

---

## 五、 Capella 高级技巧与最佳实践

### 1. 建模规则校验（Validation Rules）

Capella 提供了强大的内置一致性检查：

* 右键点击项目根节点 -> 选择  **Validate Model** 。
* 检查器会提示诸如“功能未分配给组件”、“组件端口缺少对应流”、“物理接口与逻辑接口不吻合”等语义错误，保障架构严谨性。

### 2. 常用捷径与快捷键

* **F2** ：快速重命名选中元素。
* **Ctrl + Shift + R** ：快速搜索全局模型元素（Find Element）。
* **RClick -> Show/Hide Elements** ：在复杂图表中显示或隐藏特定的端口、连接线或自功能，保持图表清晰。

### 3. Git 版本控制配合

Capella 模型文件为 XML 格式（`.melodymodeller`）。在进行 Git 版本管理时建议配置 `.gitignore`：

```
# Capella local user settings
*.aird.state
.project
/bin/
```

对于团队协同，建议尽量使用分支开发，避免同时多人在同一个 `.melodymodeller` 主文件中进行大规模交叉重构。

---

## 六、 总结与后续学习路径

掌握 Capella 的关键在于 **思想先行（Arcadia）** ，而非仅仅记住菜单位置。

null.  **第一阶段** ：熟练利用 `Activity Explorer` 完成 SA -> LA -> PA 的基本功能链与组件分配。
null.  **第二阶段** ：掌握数据建模（Data Modeling）、状态机与序列图（Scenario）绘制。
null.  **第三阶段** ：引入 **M2Doc** 自动输出标准化 Word 研发文档，并将架构成果通过 Python4Capella 集成至 CI/CD 流水线。
