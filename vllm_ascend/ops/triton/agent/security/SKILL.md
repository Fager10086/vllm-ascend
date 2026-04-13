# 多语言安全代码审查 (Security Code Review)

## 适用场景 (When to use this skill)

- **安全审查**: 对 Python、C++、Shell、Markdown 文件进行安全代码审查
- **Code Review**: 在代码评审中检查安全漏洞
- **新代码编写**: 编写安全的代码，避免常见漏洞
- **合规检查**: 满足安全合规要求 (CWE, CERT, OWASP)
- **CI/CD 集成**: 在流水线中集成安全扫描工具

---

## Python 安全审查

### 1. 代码注入

```python
# ❌ 不安全：eval/exec 执行用户输入
user_input = request.args.get("expr")
result = eval(user_input)  # 任意代码执行

# ✅ 安全：使用 ast.literal_eval 或白名单
import ast
result = ast.literal_eval(user_input)  # 仅解析字面量
```

```python
# ❌ 不安全：subprocess 使用 shell=True
import subprocess
subprocess.run(f"grep {user_input} /var/log/app.log", shell=True)  # 命令注入

# ✅ 安全：使用列表参数，避免 shell=True
subprocess.run(["grep", user_input, "/var/log/app.log"], shell=False)
```

### 2. SQL 注入

```python
# ❌ 不安全：字符串拼接 SQL
cursor.execute(f"SELECT * FROM users WHERE name = '{name}'")

# ✅ 安全：参数化查询
cursor.execute("SELECT * FROM users WHERE name = %s", (name,))
```

### 3. 反序列化漏洞

```python
# ❌ 不安全：pickle 加载不受信任的数据
import pickle
data = pickle.loads(untrusted_bytes)  # 任意代码执行

# ❌ 不安全：yaml.load 无 Loader
import yaml
config = yaml.load(untrusted_yaml)  # 任意代码执行

# ✅ 安全：使用 safe_load
config = yaml.safe_load(untrusted_yaml)

# ✅ 安全：使用 JSON 替代 pickle
import json
data = json.loads(untrusted_string)
```

### 4. 路径遍历

```python
# ❌ 不安全：直接拼接用户输入的路径
file_path = os.path.join("/data/uploads", user_filename)
with open(file_path) as f:  # ../../../etc/passwd
    content = f.read()

# ✅ 安全：验证解析后的路径在允许范围内
import os
base_dir = os.path.realpath("/data/uploads")
file_path = os.path.realpath(os.path.join(base_dir, user_filename))
if not file_path.startswith(base_dir):
    raise ValueError("Path traversal detected")
with open(file_path) as f:
    content = f.read()
```

### 5. 敏感信息泄露

```python
# ❌ 不安全：硬编码密钥
API_KEY = "sk-1234567890abcdef"
DB_PASSWORD = "admin123"

# ❌ 不安全：日志中记录敏感信息
logger.info(f"User login: password={password}")
logger.debug(f"API response: {response.json()}")  # 可能含敏感数据

# ✅ 安全：从环境变量读取
import os
API_KEY = os.environ["API_KEY"]
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is required")

# ✅ 安全：日志脱敏
logger.info(f"User login: user={username}")
logger.debug(f"API response status: {response.status_code}")
```

### 6. assert 误用

```python
# ❌ 不安全：用 assert 做运行时检查（python -O 会跳过 assert）
assert user.is_admin, "Unauthorized"

# ✅ 安全：用显式条件判断
if not user.is_admin:
    raise PermissionError("Unauthorized")
```

### 7. 临时文件安全

```python
# ❌ 不安全：可预测的临时文件名
with open("/tmp/myapp_data.txt", "w") as f:  # 竞态条件 + 符号链接攻击
    f.write(data)

# ✅ 安全：使用 tempfile
import tempfile
with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
    f.write(data)
```

### 8. 正则表达式拒绝服务 (ReDoS)

```python
# ❌ 不安全：嵌套量词导致指数级回溯
import re
pattern = re.compile(r"(a+)+$")  # ReDoS
pattern.match("a" * 30 + "!")  # 极慢

# ✅ 安全：使用原子组或限制输入长度
pattern = re.compile(r"a+$")
if len(user_input) > 1000:
    raise ValueError("Input too long")
```

### Python 安全工具

| 工具 | 用途 | 命令 |
|------|------|------|
| **bandit** | 静态安全分析 | `bandit -r src/` |
| **safety / pip-audit** | 依赖漏洞扫描 | `pip-audit` |
| **pylint** | 代码质量 + 部分安全规则 | `pylint src/` |
| **mypy** | 类型检查，防止类型混淆 | `mypy src/` |
| **semgrep** | 自定义安全规则 | `semgrep --config=p/python` |

---

## C++ 安全审查

### 1. 缓冲区溢出

```cpp
// ❌ 不安全：未检查边界
char buf[64];
strcpy(buf, user_input);  // 缓冲区溢出
sprintf(buf, "Hello %s", user_input);  // 同样不安全

// ✅ 安全：使用安全函数或 std::string
std::string buf(user_input);  // 自动管理内存

// 如果必须用 C 字符串：
char buf[64];
strncpy(buf, user_input, sizeof(buf) - 1);
buf[sizeof(buf) - 1] = '\0';
snprintf(buf, sizeof(buf), "Hello %s", user_input);
```

### 2. 内存管理

```cpp
// ❌ 不安全：裸指针 + 手动内存管理
int* ptr = new int[100];
// ... 异常发生 → 内存泄漏
delete[] ptr;  // 可能不会执行

// ❌ 不安全：use-after-free
int* p = new int(42);
delete p;
*p = 10;  // 未定义行为

// ❌ 不安全：double-free
delete p;
delete p;  // 未定义行为

// ✅ 安全：使用智能指针
auto ptr = std::make_unique<int[]>(100);  // 自动释放
auto shared = std::make_shared<MyClass>();  // 共享所有权
```

### 3. 整数溢出

```cpp
// ❌ 不安全：未检查整数溢出
int size = get_user_size();  // 可能为负数或极大值
char* buf = new char[size];  // 整数溢出 → 分配小缓冲区

// ✅ 安全：检查范围
size_t size = get_user_size();
if (size == 0 || size > MAX_ALLOWED_SIZE) {
    throw std::invalid_argument("Invalid size");
}
auto buf = std::make_unique<char[]>(size);
```

### 4. 格式化字符串漏洞

```cpp
// ❌ 不安全：用户控制格式字符串
printf(user_input);  // 格式化字符串攻击，可读写内存
fprintf(stderr, user_input);

// ✅ 安全：始终使用固定格式字符串
printf("%s", user_input);
fprintf(stderr, "%s", user_input);

// ✅ 更好：使用 C++ 流
std::cout << user_input << std::endl;
// 或 C++20 std::format
std::string msg = std::format("User: {}", user_input);
```

### 5. 未初始化变量

```cpp
// ❌ 不安全：未初始化变量
int status;  // 未初始化
if (condition) {
    status = 0;
}
return status;  // 未初始化时为未定义行为

// ✅ 安全：始终初始化
int status = -1;
if (condition) {
    status = 0;
}
return status;
```

### 6. RAII 与资源泄漏

```cpp
// ❌ 不安全：手动资源管理
FILE* fp = fopen("data.txt", "r");
// ... 如果异常发生，fp 不会被关闭
fclose(fp);

std::mutex mtx;
mtx.lock();
// ... 如果异常发生，锁不会释放
mtx.unlock();

// ✅ 安全：RAII
{
    std::ifstream file("data.txt");  // 析构时自动关闭
    // ...
}

{
    std::lock_guard<std::mutex> lock(mtx);  // 析构时自动释放
    // ...
}
```

### 7. 线程安全

```cpp
// ❌ 不安全：数据竞争
int counter = 0;
void increment() { counter++; }  // 多线程下数据竞争

// ✅ 安全：使用原子操作或锁
std::atomic<int> counter{0};
void increment() { counter.fetch_add(1); }

// 或使用互斥锁
std::mutex mtx;
int counter = 0;
void increment() {
    std::lock_guard<std::mutex> lock(mtx);
    counter++;
}
```

### 8. 类型转换安全

```cpp
// ❌ 不安全：C 风格强制转换
Base* base = (Base*)derived;  // 不安全，无类型检查
void* ptr = (void*)obj;

// ✅ 安全：使用 C++ 类型转换
auto* derived = dynamic_cast<Derived*>(base);  // 运行时类型检查
if (derived == nullptr) {
    // 处理转换失败
}
auto value = static_cast<int>(float_value);  // 明确意图
```

### C++ 安全工具

| 工具 | 用途 | 命令/用法 |
|------|------|----------|
| **AddressSanitizer** | 内存错误检测 | `-fsanitize=address` |
| **ThreadSanitizer** | 数据竞争检测 | `-fsanitize=thread` |
| **UBSanitizer** | 未定义行为检测 | `-fsanitize=undefined` |
| **Valgrind** | 内存泄漏检测 | `valgrind --leak-check=full ./app` |
| **cppcheck** | 静态分析 | `cppcheck --enable=all src/` |
| **clang-tidy** | Linter + 安全规则 | `clang-tidy -checks='*' src/*.cpp` |
| **Coverity** | 企业级静态分析 | CI 集成 |

---

## Shell 安全审查

### 1. 脚本头部安全设置

```bash
# ✅ 每个脚本必须以此开头
#!/bin/bash
set -euo pipefail
# -e: 命令失败时立即退出
# -u: 引用未定义变量时报错
# -o pipefail: 管道中任一命令失败则整体失败
```

### 2. 变量引用（最常见的 Shell 安全问题）

```bash
# ❌ 不安全：未引用变量
rm -rf $dir/$file           # 如果 $dir 为空 → rm -rf /
cp $file /backup/           # 文件名含空格时出错
if [ $var = "yes" ]; then   # $var 为空时语法错误

# ✅ 安全：始终用双引号包裹变量
rm -rf "${dir:?}/${file:?}"   # :? 确保变量非空
cp "$file" /backup/
if [ "$var" = "yes" ]; then
```

### 3. 命令注入

```bash
# ❌ 不安全：eval 执行用户输入
eval "$user_input"
eval "echo $user_data"

# ❌ 不安全：反引号中的未过滤输入
result=$(echo $user_input | grep pattern)

# ✅ 安全：避免 eval，使用参数化
result=$(grep -F -- "$user_input" "$file")
# -- 防止参数被解释为选项
# -F 固定字符串匹配（避免正则注入）
```

### 4. 临时文件安全

```bash
# ❌ 不安全：可预测的临时文件
echo "$data" > /tmp/myapp.tmp  # 竞态条件 + 符号链接攻击

# ✅ 安全：使用 mktemp
tmpfile=$(mktemp /tmp/myapp.XXXXXX)
trap 'rm -f "$tmpfile"' EXIT  # 确保退出时清理
echo "$data" > "$tmpfile"

# ✅ 安全：临时目录
tmpdir=$(mktemp -d /tmp/myapp.XXXXXX)
trap 'rm -rf "$tmpdir"' EXIT
```

### 5. 权限与文件操作

```bash
# ❌ 不安全：过于宽松的权限
chmod 777 "$file"        # 任何人都可读写执行
chmod 666 "$config"      # 任何人都可读写

# ✅ 安全：最小权限原则
chmod 750 "$script"      # 所有者可执行，组可读
chmod 640 "$config"      # 所有者可读写，组可读
chmod 600 "$secret_key"  # 仅所有者可读写

# ❌ 不安全：不检查文件是否为符号链接
if [ -f "$file" ]; then
    cat "$file"  # 可能是符号链接指向敏感文件
fi

# ✅ 安全：检查符号链接
if [ -f "$file" ] && [ ! -L "$file" ]; then
    cat "$file"
fi
```

### 6. PATH 安全

```bash
# ❌ 不安全：依赖相对路径
curl http://example.com  # 如果 PATH 被劫持，可能执行恶意 curl

# ✅ 安全：使用绝对路径（关键命令）
/usr/bin/curl http://example.com

# ✅ 安全：脚本开头设置安全 PATH
export PATH="/usr/local/bin:/usr/bin:/bin"
```

### 7. 信号处理与清理

```bash
# ✅ 安全：使用 trap 确保清理
cleanup() {
    rm -f "$tmpfile"
    # 其他清理操作
}
trap cleanup EXIT INT TERM
```

### 8. 输入验证

```bash
# ❌ 不安全：未验证输入直接使用
read -r filename
cat "$filename"  # 路径遍历

# ✅ 安全：验证输入
read -r filename
# 只允许字母数字和下划线
if [[ ! "$filename" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
    echo "Invalid filename" >&2
    exit 1
fi
# 确保在预期目录内
filepath="/data/uploads/$filename"
realpath=$(realpath "$filepath")
if [[ "$realpath" != /data/uploads/* ]]; then
    echo "Path traversal detected" >&2
    exit 1
fi
```

### Shell 安全工具

| 工具 | 用途 | 命令 |
|------|------|------|
| **ShellCheck** | Shell 脚本静态分析 | `shellcheck script.sh` |
| **shfmt** | Shell 格式化 | `shfmt -d script.sh` |

---

## Markdown 安全审查

### 1. XSS 注入（内嵌 HTML）

```markdown
<!-- ❌ 不安全：内嵌恶意 HTML/JavaScript -->
<script>alert('XSS')</script>
<img src=x onerror="alert('XSS')">
<a href="javascript:alert('XSS')">Click me</a>
<div onmouseover="steal(document.cookie)">Hover me</div>

<!-- ✅ 安全：审查时应标记所有内嵌 HTML -->
<!-- 使用安全的 Markdown 渲染器，禁用原始 HTML -->
<!-- 配置: sanitize: true 或使用 DOMPurify -->
```

### 2. 链接安全

```markdown
<!-- ❌ 不安全：javascript: 协议 -->
[Click here](javascript:alert('XSS'))

<!-- ❌ 可疑：外部链接未标注 -->
[下载工具](http://malicious-site.com/tool.exe)

<!-- ✅ 安全：仅允许 https 链接 -->
[文档](https://docs.example.com)

<!-- ✅ 审查要点：检查所有外部链接的合法性 -->
```

### 3. 敏感信息泄露

```markdown
<!-- ❌ 不安全：文档中包含真实密钥 -->
API Key: `sk-1234567890abcdef`
数据库连接: `postgresql://admin:password123@prod-db:5432/mydb`
内部服务地址: `http://10.0.1.50:8080/admin`

<!-- ✅ 安全：使用占位符 -->
API Key: `<YOUR_API_KEY>`
数据库连接: `postgresql://<user>:<password>@<host>:5432/<db>`
内部服务地址: `http://<internal-host>:<port>/admin`
```

### 4. 图片安全

```markdown
<!-- ❌ 可疑：外部图片可能追踪用户 -->
![avatar](http://tracker.evil.com/pixel.gif?user=123)

<!-- ❌ 不安全：超大图片可能导致 DoS -->
![](http://example.com/huge-100mb-image.png)

<!-- ✅ 安全：使用本地或受信任的图片源 -->
![架构图](./docs/images/architecture.png)
![Logo](https://cdn.trusted-domain.com/logo.png)
```

### Markdown 审查要点

- 搜索所有 `<script>`, `<iframe>`, `<object>`, `<embed>` 标签
- 搜索所有 `javascript:`, `data:`, `vbscript:` 协议链接
- 搜索所有硬编码的密钥、密码、Token 模式（如 `sk-`, `ghp_`, `AKIA`）
- 检查所有外部 URL 的合法性

---

## 通用安全约束 (Constraints)

### 必须遵守 (MUST)

1. **输入验证**: 验证所有外部输入，永远不信任用户数据
2. **参数化查询**: 使用参数化查询防止注入攻击
3. **最小权限**: 进程、文件、用户使用最小必要权限
4. **安全默认值**: 变量初始化、错误处理使用安全的默认值
5. **依赖审计**: 定期扫描依赖中的已知漏洞
6. **敏感信息保护**: 密钥、密码、Token 通过环境变量管理
7. **日志脱敏**: 日志中不记录密码、密钥等敏感信息
8. **错误处理**: 错误消息不泄露内部实现细节

### 禁止事项 (MUST NOT)

1. **禁止 eval 类函数**: Python `eval()`/`exec()`、Shell `eval`、C++ 无等价物但禁止动态代码生成
2. **禁止硬编码密钥**: 不在代码中硬编码密码、API Key、Token
3. **禁止提交敏感文件**: `.env`、私钥、证书不得提交到版本控制
4. **禁止忽略错误**: 不得静默吞掉异常或忽略返回值
5. **禁止过宽权限**: 不使用 `chmod 777`、`0.0.0.0` 无限制监听
6. **禁止使用已弃用的不安全函数**: `gets()`, `sprintf()`, `strcpy()` 等

---

## 安全审查检查清单 (Checklist)

### Python
```
- [ ] 无 eval()/exec() 使用不受信任的输入
- [ ] 无 pickle.loads() 加载不受信任的数据
- [ ] 使用 yaml.safe_load() 替代 yaml.load()
- [ ] subprocess 调用不使用 shell=True
- [ ] SQL 查询使用参数化方式
- [ ] 文件路径操作有路径遍历防护
- [ ] 无硬编码的密钥/密码
- [ ] 日志不记录敏感信息
- [ ] 使用 assert 不做安全检查
- [ ] 依赖已通过 pip-audit/bandit 扫描
- [ ] 正则表达式无 ReDoS 风险
- [ ] 临时文件使用 tempfile 模块
```

### C++
```
- [ ] 无缓冲区溢出风险 (strcpy → strncpy/std::string)
- [ ] 使用智能指针管理内存
- [ ] 无 use-after-free / double-free
- [ ] 整数运算有溢出检查
- [ ] printf 系列函数使用固定格式字符串
- [ ] 所有变量在使用前初始化
- [ ] 资源管理遵循 RAII 原则
- [ ] 多线程代码无数据竞争
- [ ] 使用 C++ 风格类型转换 (static_cast/dynamic_cast)
- [ ] 编译启用安全选项 (-Wall -Werror -fsanitize=address)
```

### Shell
```
- [ ] 脚本使用 set -euo pipefail
- [ ] 所有变量使用双引号包裹 ("$var")
- [ ] 无 eval 使用用户输入
- [ ] 临时文件使用 mktemp
- [ ] 文件权限不超过 755（脚本）/ 644（配置）
- [ ] 关键命令使用绝对路径
- [ ] 有 trap 清理机制
- [ ] 输入经过验证和过滤
- [ ] 通过 ShellCheck 无警告
- [ ] 不使用 . 或空目录在 PATH 中
```

### Markdown
```
- [ ] 无内嵌 <script>/<iframe> 标签
- [ ] 无 javascript:/data: 协议链接
- [ ] 无硬编码密钥/密码/Token
- [ ] 无内部 IP 地址或内部 URL 泄露
- [ ] 外部图片来源可信
- [ ] 无追踪像素
```

---

## 推荐工具汇总

| 语言 | 工具 | 类型 | 说明 |
|------|------|------|------|
| Python | **bandit** | 静态分析 | Python 安全漏洞检测 |
| Python | **pip-audit** | 依赖扫描 | Python 依赖漏洞检查 |
| Python | **semgrep** | 规则引擎 | 自定义安全规则匹配 |
| Python | **mypy** | 类型检查 | 类型安全，防止类型混淆 |
| C++ | **AddressSanitizer** | 运行时检测 | 内存错误检测 |
| C++ | **ThreadSanitizer** | 运行时检测 | 数据竞争检测 |
| C++ | **cppcheck** | 静态分析 | C/C++ 静态分析 |
| C++ | **clang-tidy** | Linter | 代码质量 + 安全规则 |
| C++ | **Valgrind** | 运行时检测 | 内存泄漏检测 |
| Shell | **ShellCheck** | 静态分析 | Shell 脚本安全分析 |
| 通用 | **git-secrets** | 预提交钩子 | 防止提交密钥 |
| 通用 | **trufflehog** | 密钥扫描 | 扫描代码中的密钥 |
| 通用 | **gitleaks** | 密钥扫描 | Git 仓库密钥泄露检测 |
| Markdown | **markdownlint** | Linter | Markdown 格式检查 |

---

## 参考标准

- [CWE Top 25](https://cwe.mitre.org/top25/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CERT C++ Secure Coding](https://wiki.sei.cmu.edu/confluence/display/cplusplus)
- [CERT C Secure Coding](https://wiki.sei.cmu.edu/confluence/display/c)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [ShellCheck Wiki](https://www.shellcheck.net/wiki/)

## Metadata

- **Version**: 1.0.0
- **Last updated**: 2026-04-11
- **Languages**: Python, C++, Shell/Bash, Markdown
- **Tags**: `#security` `#code-review` `#python` `#cpp` `#shell` `#markdown` `#OWASP` `#CWE`
