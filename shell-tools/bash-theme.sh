# 先删掉旧的那段（如果你之前加过）
# perl -0777 -i -pe 's/\n# ---- pretty PS1 with git branch.*?\n\n//s' ~/.bashrc

# 写入新的完整配置
cat >> ~/.bashrc <<'EOF'

# ---- pretty PS1 with git branch (fixed, .../last2) ----

__git_branch() {
  git rev-parse --abbrev-ref HEAD 2>/dev/null
}

# 显示路径：若层级>2，只显示最后两级并在前面加 .../
# 例：/root/projects/bailingtts_cfm -> .../projects/bailingtts_cfm
#     ~/a -> ~/a
__pwd_last2() {
  local p="${PWD/#$HOME/~}"
  awk -F/ '{
    if (NF<=2) {print $0; next}
    print ".../" $(NF-1) "/" $NF
  }' <<< "$p"
}

# PS1 里用的颜色（必须用 \[ \] 包住）
C_RESET='\[\e[0m\]'
C_GRAY='\[\e[90m\]'
C_GREEN='\[\e[32m\]'
C_BLUE='\[\e[34m\]'
C_RED='\[\e[31m\]'

# 给分支用的颜色（放在变量里时用真实 ESC，不用 \[ \]）
BR_YELLOW=$'\e[33m'
BR_RESET=$'\e[0m'

if [[ $EUID -eq 0 ]]; then
  C_USER="$C_RED"
  PROMPT_END="#"
else
  C_USER="$C_GREEN"
  PROMPT_END="$"
fi

# 每次提示符前更新：时间(含秒) + git 分支
PROMPT_COMMAND='
  __TIME=$(date +%H:%M:%S)
  __BRANCH=$(__git_branch)
  if [[ -n "$__BRANCH" ]]; then
    __BRANCH=" ${BR_YELLOW}${__BRANCH}${BR_RESET}"
  else
    __BRANCH=""
  fi
'

# [user path HH:MM:SS] branch #
PS1="${C_GRAY}[${C_USER}\u${C_RESET} ${C_BLUE}\$(__pwd_last2)${C_RESET} ${C_GRAY}\${__TIME}${C_RESET}]${C_RESET}\${__BRANCH} ${C_USER}${PROMPT_END}${C_RESET} "

EOF

# 立刻生效
source ~/.bashrc
