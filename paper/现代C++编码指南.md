# 现代 C++ 核心语法与编码实战教程

本教程旨在为开发者梳理现代 C++（C++11/14/17/20）最核心的语法特性、底层物理机制以及工业级（如高性能系统、嵌入式及车载安全关键规范）的编码最佳实践。

---

## 目录

null. [类型系统、变量与统一初始化](#1-类型系统变量与统一初始化)
null. [指针、引用与 ](#2-指针引用与---符号用法精算)`*`[ / ](#2-指针引用与---符号用法精算)`&`[ 符号用法精算](#2-指针引用与---符号用法精算)
null. [现代值语义、智能指针与移动语义](#3-现代值语义智能指针与移动语义)
null. [面向对象基石：封装、继承赋值与动态多态](#4-面向对象基石封装继承赋值与动态多态)
null. [构造函数演进与类设计最佳实践](#5-构造函数演进与类设计最佳实践)
null. [泛型编程、模板、SFINAE 与 C++20 Concepts](#6-泛型编程模板sfinae-与-c20-concepts)
null. [现代 STL 容器、算法与 Lambda 表达式深度应用](#7-现代-stl-容器算法与-lambda-表达式深度应用)

---

## 1. 类型系统、变量与统一初始化

C++ 是一门 **强类型静态语言** 。现代 C++ 极力推崇编译期安全性与零运行时开销。

### 1.1 `auto` 关键字（类型推导）

`auto` 让编译器在编译期根据变量的初始值自动推导出其真实类型，绝不影响运行时性能。

```
auto salary = 50000.0;                 // 推导为 double
const auto& name = std::string("C++"); // 推导为 const std::string&，避免拷贝
```

### 1.2 `size_t` 类型

`size_t` 是系统定义的无符号整数类型，保证能安全表示当前平台上任何对象或内存的大小：

* **32位系统** ：等价于 `unsigned int`（4 字节）。
* **64位系统** ：等价于 `unsigned long long`（8 字节）。
* **最佳实践** ：在任何涉及数组下标、容器大小（如 `vector.size()`）或内存字节数时，**一律使用 **`size_t`，避免使用带符号的 `int` 导致负数越界或隐式类型提升错误。

### 1.3 列表初始化与防止窄化转换

现代 C++ 推荐使用大括号 `{}` 进行统一初始化（Uniform Initialization）。相比传统圆括号 `()`，大括号初始化会在编译期强制拦截 **窄化转换（Narrowing Conversion）** ：

```
double pi = 3.14159;

int x(pi); // ❌ 传统写法：隐式将 double 截断为 int (值为3)，编译器保持沉默
int y{pi}; // 🟢 现代写法：编译器直接抛出 Error/Warning，拒绝精度丢失的窄化转换
```

---

## 2. 指针、引用与 `*` / `&` 符号用法精算

C++ 赋予了开发者直接操作物理内存的能力。理解 `*` 和 `&` 在不同上下文中的双重身份至关重要。

### 2.1 区分 `*` 与 `&` 的身份卡片

| 符号  | 场景                 | 语法角色               | 物理意义                                         | 示例              |
| ----- | -------------------- | ---------------------- | ------------------------------------------------ | ----------------- |
| `*` | **声明变量时** | **指针定义符**   | 声明该变量保存的是一个**内存地址**         | `int* p = &a;`  |
|       | **使用变量时** | **解引用运算符** | 顺着地址，**取出**对应内存里的值           | `*p = 20;`      |
| `&` | **声明变量时** | **引用定义符**   | 声明该变量是已有变量的**绑定别名（外号）** | `int& ref = a;` |
|       | **使用变量时** | **取地址运算符** | **获取**一个变量在内存中的物理地址         | `p = &a;`       |

* **黄金判定法则** ：看符号左边是否有数据类型。有类型（如 `char*`、`float&`）代表声明指针或引用；无类型（如 `*ptr`、`&val`）代表解引用或取地址。

### 2.2 常量指针（`const T*`）与 指针常量（`T* const`）

通过**“从右往左读（倒读法）”**轻松区分：

```
const int* p1; // 1. 倒读：p1 is a pointer to constant int.
               //    物理意义：指向的内容只读，但指针本身指向可变（常量指针）。

int* const p2 = &x; // 2. 倒读：p2 is a constant pointer to int.
                    //    物理意义：指针本身只读，不能指向别处；但指向的内容可变（指针常量）。
```

### 2.3 指针与引用的物理区别

* **指针** 是一个独立的变量，存储内存地址，在 64 位系统上占 8 字节。可以为 `nullptr`，随时可重定向（变心）。
* **引用** 是已有对象的别名，在语法上不占独立内存。 **绝不可为空** ，且一经初始化绑定便终身不可改绑。

---

## 3. 现代值语义、智能指针与移动语义

手动 `new` 和 `delete` 是 C++ 内存泄漏与野指针的万恶之源。现代 C++ 引入 RAII（资源获取即初始化）和智能指针彻底解决这一痛点。

### 3.1 智能指针三剑客 (`<memory>`)

null. `std::unique_ptr` **（独占型）** ：
      * 同一时间只能有一个指针拥有该堆内存。禁止拷贝，只能通过 `std::move` 转移所有权。
      *  **性能最高** ：开销几乎为零。首选。
null. `std::shared_ptr` **（共享型）** ：
      * 采用引用计数机制，允许多个指针共享同一块内存。当计数降为 0 时自动释放内存。
null. `std::weak_ptr` **（弱引用）** ：
      * 辅助 `shared_ptr`。不增加引用计数，用以观察资源是否存在，专门用来 **解决循环引用导致的内存泄漏灾难** 。

```
// 循环引用灾难解决示例
class Node {
public:
    std::string m_name;
    std::weak_ptr<Node> neighbor; // 💡 使用 weak_ptr 替代 shared_ptr 避开死锁循环
    Node(std::string name) : m_name(name) {}
    ~Node() { std::cout << m_name << " Destroyed\n"; }
};
```

### 3.2 右值引用 (`&&`) 与 移动语义 (`std::move`)

* **右值引用 (** `&&`**)** 专门用于绑定即将消亡的临时对象（右值）。
* **移动语义** 允许我们“夺取”临时对象的底层指针/资源，而无需进行昂贵的深拷贝，实现零成本性能跃升。

```
std::vector<std::string> vec;
std::string str = "Very Large Data String...";

// vec 直接接管 str 原本的堆内存指针，str 变为空。零拷贝开销，极快！
vec.push_back(std::move(str)); 
```

---

## 4. 面向对象基石：封装、继承赋值与动态多态

### 4.1 封装（Encapsulation）

利用访问修饰符（`private`、`protected`、`public`）隐藏内部实现细节，暴露出安全且高内聚的公共接口。

```
class BankAccount {
private:
    double m_balance = 0.0; // 私有属性，防随意篡改
public:
    void deposit(double amount) { // 安全控制的大门
        if (amount > 0) m_balance += amount;
    }
    double getBalance() const { return m_balance; } // 只读接口
};
```

### 4.2 继承中的赋值规则与“对象切片（Object Slicing）”

当父类和子类发生值赋值、指针或引用赋值时，遵循以下底层物理规则：

```
class Base { public: virtual void print() { std::cout << "Base\n"; } };
class Derived : public Base { public: void print() override { std::cout << "Derived\n"; } };
```

null. `Base* p = new Derived();`**（堆内存 + 父类指针）**
      *  **物理行为** ：在堆上创建了完整的 `Derived` 对象，`p` 是指向它的父类指针。
      *  **多态性** ： **有** 。调用 `p->print()` 查虚函数表，执行子类版本。
null. `Base ptr = Derived();`**（栈内存 + 值对象赋值 🚨 对象切片）**
      *  **物理行为** ：在栈上强行分配一个 `Base` 大小的对象空间。子类特有的数据和虚表指针被 **无情切掉（Slicing）** ，仅拷贝了父类部分的数据。
      *  **多态性** ： **无** 。`ptr` 已彻底退化为普通父类对象。
null. `Derived* p = new Base();`**（向下转型 ❌ 语法错误）**
      *  **物理行为** ：禁止！因为父类内存空间不包含子类特有的成员，若强行通过子类指针去访问父类，会导致致命的内存越界。

### 4.3 动态多态与虚函数表（vtable）

多态公式：

$$
\text{动态多态} = \text{继承} + \text{虚函数(virtual)} + \text{函数重写(override)} + \text{父类指针/引用指向子类对象}
$$


* **重载（Overload）** ：同一个作用域内，函数名相同但 **参数不同** 。在编译期静态决定。
* **重写（Override）** ：子类重新实现父类中签名**完全相同**的虚函数。在运行期通过查询虚表（vtable）和虚表指针（vptr）实现动态绑定。
* **名字隐藏（Name Hiding）** ：若在子类中重写了同名但参数不同的函数，且未使用 `using` 引入，父类的同名函数会被完全隐藏，无法通过子类对象直接调用。

---

## 5. 构造函数演进与类设计最佳实践

### 5.1 初始化列表的物理过程

为什么推荐在构造函数中使用初始化列表？

```
// 写法 A（初始化列表）
Controller(const std::string& name) : m_name(name) {}

// 写法 B（{} 内部赋值）
Controller(const std::string& name) { m_name = name; }
```

* **写法 A（一步到位）** ：直接调用 `std::string` 的拷贝构造函数，在 `m_name` 诞生的瞬间完成初始化。
* **写法 B（二次操作）** ：在进入 `{}` 前，编译器先调用 `std::string` 的默认构造函数将 `m_name` 初始化为空串 `""`；进入 `{}` 后，再调用 `operator=` 擦除空串、赋值新内容。带来了无谓的开销。
* **语法强制性** ：若类成员包含 `const` ** 常量** 、 **引用（** `&` **）** 、或 **无默认构造函数的第三方类** ，**必须**使用初始化列表进行初始化，否则编译直接报错。

### 5.2 现代化便捷构造方案

```
class Task {
private:
    int m_id{0};                           // 1. 类内默认初始化 (C++11 强烈推荐)
    std::string m_status{"Pending"};

public:
    Task(int id, const std::string& status) : m_id(id), m_status(status) {}

    // 2. 委托构造函数：避免多重载初始化代码冗余
    Task(int id) : Task(id, "Pending") {}  

    // 3. 显式缺省与禁用 (生命周期掌控)
    Task() = default;                      // 显式生成默认构造
    Task(const Task&) = delete;            // 显式禁用拷贝构造（防止驱动、句柄被误复制）
    Task& operator=(const Task&) = delete; 
};
```

---

## 6. 泛型编程、模板、SFINAE 与 C++20 Concepts

泛型编程旨在编写与类型无关的通用代码，提供绝对的 **编译期静态多态** ，实现零运行时开销。

### 6.1 模板实例化底层机制

模板（如 `template <typename T>`）在编译前并不是真实的机器码，只是一张“设计图纸”。只有当你在代码中具体调用时，编译器才会在**编译期**根据传入的实际类型，自动特化生成一份真实的代码（特化生成）。
因此，C++ 模板不会像 Java 泛型那样在运行时发生“类型擦除”和装箱拆箱开销。

```
template <typename T, size_t N>
class SafeArray {
private:
    T m_data[N]; // 非类型模板参数：直接在栈上分配固定大小，极其契合嵌入式安全规范
};
```

### 6.2 替换失败非错误（SFINAE）与 C++20 Concepts

传统的模板偏特化在遇到不匹配的类型时，常常产生晦涩难懂的数页报错。

* **SFINAE (Substitution Failure Is Not An Error)** ：
  如果在推导模板签名时发生类型不匹配，编译器不视其为错误，而是优雅地将该模板移出候选名单，转去寻找其他可用的重载函数。
* **C++20 Concepts（概念约束）** ：
  现代 C++ 使用 `requires` 关键字限制模板参数类型，将复杂的类型检查直接锁死在接口签名处。

```
#include <concepts>

// 限制模板参数 T 必须支持迭代器操作（即必须是容器类型）
template <typename T>
requires requires(T t) { t.begin(); t.end(); }
void processContainer(const T& container) {
    std::cout << "Container version\n";
}

void processContainer(double scalar) {
    std::cout << "Scalar double version\n";
}
```

---

## 7. 现代 STL 容器、算法与 Lambda 表达式深度应用

### 7.1 标准库算法与 Lambda 表达式

现代 C++ 的核心原则之一是：**尽量不要手写 **`for` ** 循环，优先使用标准算法（** `<algorithm>` **）** 。

```
std::vector<int> nums = {4, 1, 3, 5, 2};

// 💡 传递 Lambda 作为比较谓词实现降序。
// 黄金法则：你想让什么样的元素排在前面，就在什么情况下让 Lambda 返回 true。
std::sort(nums.begin(), nums.end(), [](int a, int b) {
    return a > b; // 我们希望大数在左，所以当第一参数 a > 较后参数 b 时返回 true
});
```

### 7.2 高效去重：Erase-Remove 惯用法

标准库算法（如 `std::unique`、`std::remove`）只是全局函数，只有迭代器视界， **无权改变容器大小** 。
它们的工作是把垃圾甩到容器尾部，并返回分界线迭代器。要真正缩减容器物理内存，必须配合容器的成员函数 `erase`。

```
std::vector<int> signal = {1, 1, 2, 2, 3, 1, 4};

// 1. unique 整理相邻重复数据，将垃圾推至末尾，并返回第一个垃圾数据的迭代器
auto trash_begin = std::unique(signal.begin(), signal.end());

// 2. 容器成员函数 erase 彻底销毁垃圾，缩减物理大小
signal.erase(trash_begin, signal.end()); // 整理后为: {1, 2, 3, 1, 4}
```

### 7.3 数组遍历的高效选型

对于数组或容器的遍历，推荐按下图所示的选型优先级进行决策：

```
                    【遍历需求选择】
                           │
             ┌─────────────┴─────────────┐
        需要控制下标?                 不需要控制下标?
             │                           │
     ┌───────┴───────┐           ┌───────┴───────┐
    是              否          只读?           需要写修改?
     │               │           │               │
  【cbegin】     【Range-for】 【const auto&】   【auto&】
  迭代器控制     基于范围循环     高效、只读      原地安全修改
```
