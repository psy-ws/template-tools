# 先删掉旧的那段（如果你之前加过）
perl -0777 -i -pe 's/\n# ---- pretty PS1 with git branch.*?\n\n//s' ~/.bashrc

# 写入新的完整配置（路径紫色）
cat >> ~/.bashrc <<'EOF'

# ---- pretty PS1 with git branch (colors customized) ----

__git_branch() {
  git rev-parse --abbrev-ref HEAD 2>/dev/null
}

# 显示路径：若层级>2，只显示最后两级并在前面加 .../
__pwd_last2() {
  local p="${PWD/#$HOME/~}"
  awk -F/ '{
    if (NF<=2) {print $0; next}
    print ".../" $(NF-1) "/" $NF
  }' <<< "$p"
}

# PS1 里用的颜色（必须用 \[ \] 包住）
C_RESET='\[\e[0m\]'
C_GRAY='\[\e[90m\]'     # 用户名灰色
C_PURPLE='\[\e[35m\]'   # 路径紫色
C_RED='\[\e[31m\]'      # []红色
C_YELLOW='\[\e[33m\]'   # 分支黄色
C_GREEN='\[\e[32m\]'    # 时间绿色
C_BLUE='\[\e[34m\]'     # # 蓝色

# 给分支用的颜色（放在变量里：用真实 ESC，不用 \[ \]）
BR_YELLOW=$'\e[33m'
BR_RESET=$'\e[0m'

PROMPT_COMMAND='
  __TIME=$(date +%H:%M:%S)
  __BRANCH=$(__git_branch)
  if [[ -n "$__BRANCH" ]]; then
    __BRANCH=" ${BR_YELLOW}${__BRANCH}${BR_RESET}"
  else
    __BRANCH=""
  fi
'

# [] 红色；root 灰色；路径 紫色；时间 绿色；分支 黄色；最后 # 蓝色
PS1="${C_RED}[${C_RESET}${C_GRAY}\u${C_RESET} ${C_PURPLE}\$(__pwd_last2)${C_RESET} ${C_GREEN}\${__TIME}${C_RESET}${C_RED}]${C_RESET}${C_YELLOW}\${__BRANCH}${C_RESET} ${C_BLUE}#${C_RESET} "

EOF

source ~/.bashrc
