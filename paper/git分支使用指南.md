# 商业项目 Git 规范与命令行实践指南

本指南旨在为商业项目中的开发者提供一套符合企业级标准的 Git 命令行操作规范与工作流。通过标准化每个开发阶段的行为，确保代码库的历史线性整洁、可追溯，并最大程度避免协作冲突。

---

## 🗺️ 整体开发生命周期拓扑

在标准的 **Feature Branching** 工作流中，一个功能的完整生命周期如下：

```
[develop] ──────────(1. pull 最新)─────────────────────────────────────────────(7. Squash Merge)──► [develop]
             \                                                                   /
              └──► [feature/ticket_desc] ──(3. commit)──► (5. rebase) ──► (6. PR/评审)
```

---

## 🚀 阶段一：准备与基线同步 (Base Synchronization)

在开始任何新功能或修复 Bug 之前，必须确保你的本地开发主干与远程仓库完全同步。

### 📌 阶段规范

null.  **禁止在旧代码上开发** ：绝不能直接在落后的本地分支上切出新分支，这会导致后续合并时面临灾难性的冲突。
null.  **本地拉取优于网页创建** ：尽量在本地拉取基线后创建分支，避免直接在 GitHub/GitLab 网页端点击创建分支，防止在远程产生大量无用分支指针。

### 🛠️ 核心命令行

```
# 1. 切换到本地开发主干（通常是 develop 或 main）
git checkout develop

# 2. 从远程获取最新代码并合并
git pull origin develop

# 3. 清理远程已被删除、但本地仍在缓存的“幽灵”分支指针
git fetch --prune
```

---

## 🌿 阶段二：创建并关联功能分支 (Branching)

所有的开发工作必须在独立的分支中进行，严禁直接在主干分支（`main`/`develop`）进行任何提交。

### 📌 分支命名规范

分支名称必须以任务类型开头，结构为：`{type}/{ticket_id}_{short_description}`

* **新功能 (Feature)** : `feature/BTS-1024_user-login`（BTS-1024 为 Jira 或管理系统的 Ticket ID）
* **缺陷修复 (Bugfix)** : `bugfix/BTS-1025_fix-jwt-expiration`
* **紧急线上修复 (Hotfix)** : `hotfix/BTS-1026_crash-on-refresh`

### 🛠️ 核心命令行

```
# 1. 基于最新的 develop 分支，本地创建并直接切换至新分支
git checkout -b feature/BTS-1024_user-login develop
# (在现代 Git 中，更推荐使用语义更明确的 switch 命令)
# git switch -c feature/BTS-1024_user-login develop

# 2. 首次推送时，将本地分支与远程关联（-u/--set-upstream 建立追踪关系）
git push -u origin feature/BTS-1024_user-login
```

---

## 🛠️ 阶段三：本地编码、暂存与挂起 (Coding & Staging)

在编码过程中，应遵循“小步快跑、频繁提交”的原则，同时确保敏感和无关信息不被提交。

### 📌 阶段规范

null.  **单次提交高内聚** ：一个 Commit 应该只解决一个问题（如：写好了一个接口，或者重构了一个类）。
null.  **严格的忽略规则** ：确保本地配置文件（如 `.env`）、本地编译产物（如 `build/`、`*.o`）以及 IDE 缓存（如 `.vscode/`）被记录在 `.gitignore` 中，严禁提交到远程。
null.  **安全暂存** ：开发进行到一半需要去修复紧急 bug 时，严禁为了切换分支而进行“临时胡乱 Commit”，应使用 `stash`。

### 🛠️ 核心命令行

```
# 1. 随时查看当前工作区和暂存区的状态
git status

# 2. 对比当前工作区修改与暂存区的具体差异
git diff

# 3. 将修改的文件添加至暂存区（精准添加，避免 git add . 引入垃圾文件）
git add src/auth/jwt_verification.cpp include/auth/jwt_verification.h

# 4. 【中断救急】暂存当前未提交的工作区和暂存区修改
git stash push -m "WIP: BTS-1024 implementing jwt token generation"

# 5. 【中断恢复】修复完紧急 bug 后，切回功能分支并恢复暂存代码
git stash pop
```

---

## ✍️ 阶段四：规范化本地提交 (Standardized Commits)

代码提交信息（Commit Message）是项目历史可追溯性的灵魂。商业项目推荐采用  **Angular Commit 规范** 。

### 📌 提交规范格式

```
<type>(<scope>): <subject>

<body>
```

* **Type (类型)** :
* `feat`: 引入新功能
* `fix`: 修复 Bug
* `docs`: 仅文档变更
* `style`: 代码格式调整（不影响运行逻辑，如缩进、空格）
* `refactor`: 重构（既不修复 bug 也不添加新功能）
* `test`: 增加或修改测试代码
* `chore`: 构建过程或辅助工具的变动（如修改 CMakeLists.txt、.gitignore）
* **Scope (范围)** : 说明修改影响的模块（如：`auth`, `database`, `ui`）。
* **Subject (主题)** : 简短描述，使用祈使句（如 `add...` 而非 `added...`），句末不加句号。

### 🛠️ 核心命令行

```
# 1. 提交暂存区代码并撰写符合规范的提交信息
git commit -m "feat(auth): add JWT verification for user login"

# 2. 【救急】如果刚刚 commit 完发现有拼写错误或遗漏了某个文件（且尚未 push 到远程）
git add src/auth/jwt_verification.cpp
git commit --amend --no-edit  # 使用 --no-edit 表示沿用上一次的 commit message
```

---

## 🔄 阶段五：同步主干与历史整理 (Rebase & Squash)

在将本地代码推送到远程仓库进行评审（PR）之前，**必须**同步最新的远程开发主干，并整理本地杂乱的提交历史。这是 VS Code 难以完美覆盖、而命令行最强大的地方。

### 📌 变基（Rebase） vs 合并（Merge）

* `git merge` ** (合并)** : 会产生一个额外的“Merge commit”节点，导致历史树出现交叉、分叉，使整体提交线变得杂乱。
* `git rebase` ** (变基)** : 将你当前分支的所有本地提交“拔起来”，插到目标分支的最前方，使提交历史呈现完美的单一直线。

```
# 初始状态
      A --- B (你的 feature 分支)
     /
C --- D (远程 develop 分支，有同事的新提交)

# 1. 运行 git merge origin/develop 后的历史：
      A --- B ───┐
     /           ▼
C --- D ─────────M (产生了一个交叉的合并节点 M)

# 2. 运行 git rebase origin/develop 后的历史：
C --- D --- A' --- B' (历史是一条完美的直线，A' 和 B' 的哈希值会更新)
```

### 🛠️ 交互式变基（Interactive Rebase）整理历史

如果你在本地开发过程中产生了很多 `fix typo`, `test`, `debug again` 等临时性的、无意义的 Commit，在发起 PR 前必须将它们**压缩（Squash）**成一个干净的 Commit。

```
# 1. 从远程获取最新的主干变动
git fetch origin

# 2. 将本地分支变基到最新的远程主干上
git rebase origin/develop

# 3. 开启交互式变基，整理最近的 4 次本地提交
git rebase -i HEAD~4
```

此时会弹出一个文本编辑器，内容类似于：

```
pick a1b2c3d feat(auth): add JWT interface
pick e5f6g7h fix: fix typo in header
pick i9j0k1l test: add auth unit test
pick m3n4o5p chore: remove debug logs
```

**整理策略：** 将第一行保留为 `pick`（代表基准提交），将其余后续的临时提交前面的 `pick` 改为 `squash`（或简写为 `s`，代表并入前一个提交）：

```
pick a1b2c3d feat(auth): add JWT interface with comprehensive verification
squash e5f6g7h fix: fix typo in header
squash i9j0k1l test: add auth unit test
squash m3n4o5p chore: remove debug logs
```

保存并退出编辑器，Git 会让你重新编辑合并后的 Commit Message，编辑完成后，这 4 个零散的提交在本地历史中便会融合成 1 个完美的提交。

---

## 🚀 阶段六：安全推送与 Pull Request (Push & Review)

### 📌 阶段规范

null.  **禁止 ** `-f` ** 强推保护分支** ：绝对不能在公共保护分支（如 `main`/`develop`）上执行强制推送。
null.  **使用租约强推** ：由于你在阶段五中执行了 `rebase`（重写了本地的历史提交），如果之前已经推送过该分支，普通的 `git push` 会被远程拒绝。此时必须使用带租约的强制推送 `--force-with-lease`，它会在远程分支未被他人更新时才允许强推，这比普通的 `-f` 安全得多。

### 🛠️ 核心命令行

```
# 安全地将整理后的线性分支推送至远程，准备发起 PR
git push --force-with-lease
```

---

## 🧹 阶段七：合并后清理 (Post-Merge Cleanup)

当你的 PR 经过 Code Review 获得批准，并最终在网页端合入主分支后，该 feature 分支的使命即宣告结束。

### 📌 最佳实践：Squash and Merge

在 GitHub/GitLab 平台端，建议勾选 **"Squash and merge"** 进行合并。其底层的逻辑等同于：

```
git merge --squash feature/BTS-1024_user-login
```

它能保证开发分支上再多的中间记录，也绝对不会污染公共 `develop` 分支的历史。

### 🛠️ 合并后的本地清理命令行

```
# 1. 切换回本地主干，并同步远程最新的合并结果
git checkout develop
git pull origin develop

# 2. 安全删除已被成功合并的本地功能分支
git branch -d feature/BTS-1024_user-login

# 3. 【高阶技巧】自动删除本地所有已经被合并到当前分支的无用本地分支（排除 develop 和 main）
git branch --merged | grep -v "(^\*|master|develop)" | xargs git branch -d
```

---

## 🛡️ 商业项目必备：紧急撤销与回退救急 (Undo & Diagnostic)

在商业项目中，当代码出错需要快速回退，或者需要排查 Bug 责任人时，以下命令行是你的“救命稻草”。

### 1. 撤销本地未提交的修改

* **撤销工作区（尚未 **`git add` **）的修改** ：
  ```
  git checkout -- <file-name>  # 放弃单个文件的修改
  git checkout .               # 放弃本地所有工作区的修改
  ```
* **撤销已暂存（已经 **`git add` ** 但尚未 commit）的修改** ：
  ```
  git reset HEAD <file-name>   # 将该文件从暂存区撤回，但保留在工作区
  ```

### 2. 撤销已 Commit 但未 Push 的本地提交

* **保留修改回退（最常用）** ：回退到上一个 Commit，把你刚写的代码完好地留在暂存区，以便你修改后重新提交。

```
  git reset --soft HEAD~1
```

* **彻底放弃修改回退（慎用！）** ：彻底回滚到上一次 Commit，你刚才写的所有代码将全部消失。

```
  git reset --hard HEAD~1
```

### 3. 撤销已 Push 到远程的公共提交 (Safe Revert)

如果你的代码已经合并或推送到公共分支，**绝对严禁**使用 `git reset` 重写远程历史（这会导致同事本地历史冲突）。必须使用 `revert`，通过产生一个“相反的”新提交来中和之前的提交。

```
# 产生一个反向的全新 Commit，以此来抵消特定 Commit 引入的改动
git revert <commit-id>
git push origin develop
```

### 4. 跨分支特定代码挑选 (`cherry-pick`)

 **场景** ：你在 `feature/A` 分支上开发时，发现同事在 `feature/B` 分支上提交了一个非常通用的工具类函数（Commit ID 为 `d3b2a1c`），而你急需它，但你不想合并他整个 `feature/B` 分支。

```
# 1. 切换回你自己的功能分支
git checkout feature/BTS-1024_user-login

# 2. 精准地将同事的那一个提交“捡”到你当前的分支上
git cherry-pick d3b2a1c
```

### 5. 多人协作排查神器

* **精简的拓扑图形化提交历史** ：

```
  git log --oneline --graph --all
```

* **精确追溯代码行责任人** （查看某个文件每一行代码是谁在什么时候、哪次提交中改写的）：

```
  git blame <file-name>
```
