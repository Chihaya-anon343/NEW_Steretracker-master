### 1. 基础语法与核心概念 (Basics) - 深入解析

* **数据类型选型的考量** ：
* **`<span data-markdown-start-index="164">size_t</span>` 的重要性** ：在表示内存大小、数组索引或容器容量时，永远使用 `<span data-markdown-start-index="260">size_t</span>` 而不是 `<span data-markdown-start-index="279">int</span>`。`<span data-markdown-start-index="287">size_t</span>` 保证能容纳当前平台上最大的对象大小（在 64 位系统上是 64 位无符号整数），且不会出现负数索引这种逻辑错误。
* **固定宽度整数** ：在处理通信协议、硬件寄存器时，必须使用 `<span data-markdown-start-index="379"><cstdint></span>` 中的 `<span data-markdown-start-index="398">uint8_t</span>`, `<span data-markdown-start-index="409">int32_t</span>` 等，以保证跨平台的绝对一致性。
* **内存模型：栈 (Stack) 与 堆 (Heap)** ：
* **栈内存** ：分配和释放速度极快，生命周期由作用域 `<span data-markdown-start-index="491">{}</span>` 严格控制。在汽车或底层软件中， **优先使用栈** 。
* **堆内存** ：通过 `<span data-markdown-start-index="508">new</span>` 或 `<span data-markdown-start-index="518">malloc</span>` 分配。存在内存碎片化、分配时间不确定（非确定性行为）以及内存泄漏的风险。
* **传参最佳实践：`<span data-markdown-start-index="576">const T&</span>` vs 按值传递** ：
* 如果对象较大（如 `<span data-markdown-start-index="616">std::string</span>` 或自定义结构体）且在函数内只读， **必须使用 `<span data-markdown-start-index="693">const T&</span>`** 。按值传递会触发深拷贝，极大浪费性能。
* 如果只是基础类型（如 `<span data-markdown-start-index="707">int</span>`, `<span data-markdown-start-index="714">double</span>`），直接按值传递即可，因为引用的底层实现是指针，对于基础类型反而可能增加解引用的开销。
* **`<span data-markdown-start-index="751">std::span</span>` (C++20)** ：
* 传统 C 语言传递数组需要传指针和长度 (`<span data-markdown-start-index="834">void process(int* arr, size_t len)</span>`)。`<span data-markdown-start-index="874">std::span</span>` 提供了一个安全、轻量级的连续内存“视图”，不拥有数据，拷贝成本极低：`<span data-markdown-start-index="988">void process(std::span<int> data)</span>`。

### 2. 面向对象编程 (OOP) - 进阶设计

* **类 (Class) 与 不变性 (Invariants)** ：
* `<span data-markdown-start-index="995">class</span>` 不仅仅是数据的集合，它的存在是为了维护某种逻辑状态（例如，一个 `<span data-markdown-start-index="1097">Date</span>` 类必须保证月份在 1-12 之间）。因此，成员变量必须是 `<span data-markdown-start-index="1177">private</span>`，所有的状态修改必须通过成员函数（如 `<span data-markdown-start-index="1241">setMonth()</span>`）进行校验。
* `<span data-markdown-start-index="1125">struct</span>` 退化为纯数据载体（POD - Plain Old Data），通常只有 `<span data-markdown-start-index="1200">public</span>` 成员，不需要复杂的构造逻辑。
* **核心准则：Rule of Three / Five / Zero** ：
* **Rule of Zero** ：最高境界。如果你的类只管理基础类型或本身已经妥善管理资源的成员（如 `<span data-markdown-start-index="1361">std::vector</span>`），你不应该手动编写析构函数、拷贝/移动构造函数和赋值运算符，让编译器自动生成。
* **Rule of Five** ：如果你手动管理了裸资源（如裸指针、文件句柄），你**必须**同时显式定义这 5 个特殊函数（析构、拷贝构造、拷贝赋值、移动构造、移动赋值），否则极易导致浅拷贝带来的“Double Free”崩溃。
* **组合优于继承 (Composition over Inheritance)** ：
* 深层的继承树（超过 3 层）会导致代码极其难以调试。优先考虑将功能封装为独立的类，并在新类中将其作为成员变量包含进来（组合），结合接口（纯虚函数类）实现多态。

### 3. 标准模板库 (STL) 与底层约束

* **STL 容器在严苛环境（如汽车电子/嵌入式）的局限** ：
* 像 `<span data-markdown-start-index="1674">std::vector</span>` 和 `<span data-markdown-start-index="1692">std::string</span>` 在容量不足时，会在堆上重新分配一块更大的内存，拷贝旧数据，再释放旧内存。这种**动态内存分配**在某些安全关键 (Safety-Critical) 的嵌入式项目中是被**严格禁止**的。
* **替代方案** ：使用固定大小的容器，如 `<span data-markdown-start-index="1856">std::array</span>`。如果需要类似 vector 的行为，通常会使用内部自研的定长容器，或者引入开源的 ETL (Embedded Template Library)，它提供了像 `<span data-markdown-start-index="2026">etl::vector<T, MAX_SIZE></span>` 这种基于栈/预分配内存的替代品。
* **算法 (`<span data-markdown-start-index="1973"><algorithm></span>`) 的威力** ：
* 永远不要写原生的 `<span data-markdown-start-index="2023">for</span>` 循环去寻找一个元素，使用 `<span data-markdown-start-index="2066">std::find_if</span>`。这不仅是因为 STL 算法经过了极致优化，更是为了 **代码的表达力** 。看到 `<span data-markdown-start-index="2180">std::find_if</span>`，维护者立刻就知道这行代码的目的是“查找”，而看到一个原始的 `<span data-markdown-start-index="2285">for</span>` 循环，则需要逐行阅读里面的逻辑才能理解。

### 4. 现代 C++ 特性 (Modern C++) - 安全与性能

* **智能指针 (Smart Pointers) 的所有权语义** ：
* `<span data-markdown-start-index="2235">std::unique_ptr</span>`：**独占**所有权。它意味着“我拥有这块内存，我销毁时内存销毁”。零性能开销，绝对优先使用。
* `<span data-markdown-start-index="2307">std::shared_ptr</span>`：**共享**所有权。内部使用引用计数，当计数为 0 时销毁。它包含原子操作，有一定性能开销。只在确实有多个对象需要共享同一块生命周期不确定的数据时使用。
* **`<span data-markdown-start-index="2407">constexpr</span>` 的编译期魔法** ：
* `<span data-markdown-start-index="2436">const</span>` 表示运行时不可变，而 `<span data-markdown-start-index="2475">constexpr</span>` 表示 **在编译期就可以计算出结果** 。将复杂的数学常数计算标记为 `<span data-markdown-start-index="2576">constexpr</span>`，可以把运行时的计算时间转移到编译期，实现真正的“零成本抽象”。
* **`<span data-markdown-start-index="2549">auto</span>` 的正确用法** ：
* `<span data-markdown-start-index="2572">auto</span>` 不是动态类型（如 Python），它是在编译期自动推导出的 **强类型** 。
* 正确用法：`<span data-markdown-start-index="2639">auto it = my_map.find(key);</span>`（避免写出极其冗长的迭代器类型）。
* 危险陷阱：不要滥用 `<span data-markdown-start-index="2711">auto</span>` 导致代码失去自解释性（例如 `<span data-markdown-start-index="2758">auto result = calculate();</span>` 让人不知道返回的是 int、double 还是一个结构体）。

### 5. 编码规范与工程化实践 - 工业级标准

* **头文件污染与 `<span data-markdown-start-index="2825">using namespace</span>`** ：
* **绝对禁止**在 `<span data-markdown-start-index="2861">.h</span>` 或 `<span data-markdown-start-index="2870">.hpp</span>` 文件的全局作用域写 `<span data-markdown-start-index="2905">using namespace std;</span>`。一旦有其他文件 include 了这个头文件，`<span data-markdown-start-index="2981">std</span>` 命名空间里的所有符号（几千个）都会被强行拉入该文件的全局作用域，极易引发命名冲突（ODR 违反）。应该在 `<span data-markdown-start-index="3136">.cpp</span>` 文件中使用，或者在局部函数作用域内使用。
* **编译器警告即错误 (`<span data-markdown-start-index="3039">-Werror</span>` / `<span data-markdown-start-index="3051">/WX</span>`)** ：
* 工业级项目通常要求编译过程“零警告”。诸如隐式类型转换截断（`<span data-markdown-start-index="3142">double</span>` 转 `<span data-markdown-start-index="3155">int</span>`）、变量未使用、控制流可能没有返回值等警告，往往是潜伏的 Bug。将警告视为错误是保证代码质量的第一道防线。
* **单元测试 (Unit Testing)** ：
* C++ 代码极度依赖测试。学习并熟练使用 **gTest (Google Test)** 等框架。良好的 C++ 代码应该是“可测试的”，这意味着需要合理的模块拆分（解耦）和依赖注入，以便通过 Mock 对象来隔离测试目标。
